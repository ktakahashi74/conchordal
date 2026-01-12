use std::fs;
use std::sync::{Arc, Mutex};

use rhai::{
    Dynamic, Engine, EvalAltResult, EvalContext, Expression, FLOAT, INT, Map, NativeCallContext,
    Position,
};

use super::api::{pop_time, push_time, set_time, spawn_min_at, spawn_with_opts_at};
use super::control::{AgentPatch, PitchConstraintMode};
use super::scenario::{Action, Scenario, SceneMarker, TimedEvent};

const SCRIPT_PRELUDE: &str = r#"
__internal_debug_seed();

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
        println!("[rhai][debug] seed={seed}");
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

    pub fn spawn_with_patch(
        &mut self,
        tag: &str,
        count: i64,
        patch: Map,
        position: Position,
    ) -> Result<i64, Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        let c = count.max(0).min(u32::MAX as i64) as u32;
        let (patch_out, patch_json) = Self::parse_agent_patch(patch, position)?;
        if let Some(pitch) = patch_out.pitch.as_ref() {
            if let Some(constraint) = pitch.constraint.as_ref() {
                let mode = constraint.mode.unwrap_or(PitchConstraintMode::Free);
                if matches!(
                    mode,
                    PitchConstraintMode::Lock | PitchConstraintMode::Attractor
                ) && constraint.freq_hz.is_none()
                {
                    return Err(Box::new(EvalAltResult::ErrorRuntime(
                        "pitch.constraint.freq_hz is required when mode != free".into(),
                        position,
                    )));
                }
            }
        }
        let patch_json = Self::apply_spawn_convenience(patch_out, patch_json, position)?;
        let action = Action::Spawn {
            tag: tag.to_string(),
            count: c,
            patch: patch_json,
        };
        self.push_event(self.cursor, vec![action]);
        Ok(c as i64)
    }

    pub fn spawn_default(
        &mut self,
        tag: &str,
        count: i64,
        position: Position,
    ) -> Result<i64, Box<EvalAltResult>> {
        self.spawn_with_patch(tag, count, Map::new(), position)
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

    pub fn set_patch(
        &mut self,
        target: &str,
        patch: Map,
        position: Position,
    ) -> Result<i64, Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        let (patch_struct, patch_json) = Self::parse_agent_patch(patch, position)?;
        if patch_struct.contains_type_switch() {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "set() cannot change body.method or phonation.type; use spawn() for type selection"
                    .into(),
                position,
            )));
        }
        let action = Action::Set {
            target: target.to_string(),
            patch: patch_json,
        };
        self.push_event(self.cursor, vec![action]);
        Ok(self.estimate_match_count(target))
    }

    pub fn remove(&mut self, target: &str, position: Position) -> Result<i64, Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.push_event(
            self.cursor,
            vec![Action::Remove {
                target: target.to_string(),
            }],
        );
        Ok(self.estimate_match_count(target))
    }

    pub fn unset(
        &mut self,
        target: &str,
        path: &str,
        position: Position,
    ) -> Result<i64, Box<EvalAltResult>> {
        self.ensure_not_ended(position)?;
        self.push_event(
            self.cursor,
            vec![Action::Unset {
                target: target.to_string(),
                path: path.to_string(),
            }],
        );
        Ok(self.estimate_match_count(target))
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

    fn parse_agent_patch(
        patch: Map,
        position: Position,
    ) -> Result<(AgentPatch, serde_json::Value), Box<EvalAltResult>> {
        let raw_val = serde_json::to_value(&patch).map_err(|e| {
            Box::new(EvalAltResult::ErrorRuntime(
                format!("Error serializing AgentPatch: {e}").into(),
                position,
            ))
        })?;
        let patch_struct: AgentPatch = serde_json::from_value(raw_val).map_err(|e| {
            let debug_map = format!("{:?}", patch);
            Box::new(EvalAltResult::ErrorRuntime(
                format!("Error parsing AgentPatch: {e} (input: {debug_map})").into(),
                position,
            ))
        })?;
        let patch_json = serde_json::to_value(&patch_struct).map_err(|e| {
            Box::new(EvalAltResult::ErrorRuntime(
                format!("Error serializing AgentPatch: {e}").into(),
                position,
            ))
        })?;
        Ok((patch_struct, patch_json))
    }

    fn apply_spawn_convenience(
        patch: AgentPatch,
        patch_json: serde_json::Value,
        position: Position,
    ) -> Result<serde_json::Value, Box<EvalAltResult>> {
        let Some(pitch) = patch.pitch else {
            return Ok(patch_json);
        };
        let Some(constraint) = pitch.constraint else {
            return Ok(patch_json);
        };
        let mode = constraint.mode.unwrap_or(PitchConstraintMode::Free);
        if !matches!(mode, PitchConstraintMode::Lock) || pitch.center_hz.is_some() {
            return Ok(patch_json);
        }
        let Some(freq_hz) = constraint.freq_hz else {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "pitch.constraint.freq_hz is required when mode != free".into(),
                position,
            )));
        };
        let mut out = patch_json;
        let serde_json::Value::Object(ref mut map) = out else {
            return Ok(out);
        };
        let pitch_entry = map
            .entry("pitch".to_string())
            .or_insert_with(|| serde_json::Value::Object(serde_json::Map::new()));
        let serde_json::Value::Object(pitch_map) = pitch_entry else {
            return Ok(out);
        };
        pitch_map.insert("center_hz".to_string(), serde_json::Value::from(freq_hz));
        Ok(out)
    }

    fn estimate_match_count(&self, target: &str) -> i64 {
        let mut events: Vec<&TimedEvent> = self
            .scenario
            .events
            .iter()
            .filter(|ev| ev.time <= self.cursor + f32::EPSILON)
            .collect();
        events.sort_by(|a, b| {
            a.time
                .partial_cmp(&b.time)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.order.cmp(&b.order))
        });
        let mut tags: Vec<String> = Vec::new();
        for ev in events {
            for action in &ev.actions {
                match action {
                    Action::Spawn { tag, count, .. } => {
                        for _ in 0..*count {
                            tags.push(tag.clone());
                        }
                    }
                    Action::Remove { target } => {
                        tags.retain(|t| !matches_tag_pattern(target, t));
                    }
                    _ => {}
                }
            }
        }
        tags.into_iter()
            .filter(|t| matches_tag_pattern(target, t))
            .count()
            .try_into()
            .unwrap_or(0)
    }
}

#[derive(Clone, Copy, Debug)]
enum GlobToken {
    AnySeq,
    AnyChar,
    Literal(char),
}

fn parse_glob_pattern(pattern: &str) -> Vec<GlobToken> {
    let mut tokens = Vec::new();
    let mut chars = pattern.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '\\' => {
                if let Some(next) = chars.next() {
                    tokens.push(GlobToken::Literal(next));
                } else {
                    tokens.push(GlobToken::Literal('\\'));
                }
            }
            '*' => tokens.push(GlobToken::AnySeq),
            '?' => tokens.push(GlobToken::AnyChar),
            _ => tokens.push(GlobToken::Literal(ch)),
        }
    }
    tokens
}

fn matches_tag_pattern(pattern: &str, text: &str) -> bool {
    let tokens = parse_glob_pattern(pattern);
    let chars: Vec<char> = text.chars().collect();
    let mut dp = vec![vec![false; chars.len() + 1]; tokens.len() + 1];
    dp[0][0] = true;
    for (i, token) in tokens.iter().enumerate() {
        match token {
            GlobToken::AnySeq => {
                for j in 0..=chars.len() {
                    if dp[i][j] {
                        dp[i + 1][j] = true;
                        if j < chars.len() {
                            dp[i][j + 1] = true;
                        }
                    }
                }
            }
            GlobToken::AnyChar => {
                for j in 0..chars.len() {
                    if dp[i][j] {
                        dp[i + 1][j + 1] = true;
                    }
                }
            }
            GlobToken::Literal(ch) => {
                for j in 0..chars.len() {
                    if dp[i][j] && chars[j] == *ch {
                        dp[i + 1][j + 1] = true;
                    }
                }
            }
        }
    }
    dp[tokens.len()][chars.len()]
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
        engine.register_fn("__internal_debug", |msg: &str| {
            println!("[rhai][debug] {msg}");
        });
        let ctx_for_seed = ctx.clone();
        engine.register_fn("__internal_debug_seed", move || {
            let ctx = ctx_for_seed.lock().expect("lock script context");
            println!("[rhai][debug] seed={}", ctx.seed);
        });
        engine.register_custom_syntax_with_state_raw(
            "debug",
            |symbols, _look_ahead, _state| {
                let next = match symbols.len() {
                    0 => Some("debug"),
                    1 => Some("("),
                    2 => Some("$expr$"),
                    3 => Some(")"),
                    _ => None,
                };
                Ok(next.map(Into::into))
            },
            false,
            move |eval_ctx: &mut EvalContext, exprs: &[Expression], _state| {
                let pos = exprs[0].position();
                let msg_dyn = eval_ctx.eval_expression_tree(&exprs[0])?;
                let msg = msg_dyn.try_cast::<String>().ok_or_else(|| {
                    Box::new(EvalAltResult::ErrorRuntime(
                        "debug msg must be a string".into(),
                        pos,
                    ))
                })?;
                println!("[rhai][debug] {msg}");
                Ok(Dynamic::UNIT)
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
                let spawned = if exprs.len() >= 3 {
                    let opts_dyn = eval_ctx.eval_expression_tree(&exprs[2])?;
                    let opts = opts_dyn.try_cast::<Map>().ok_or_else(|| {
                        Box::new(EvalAltResult::ErrorRuntime(
                            "spawn patch must be a map".into(),
                            pos_opts,
                        ))
                    })?;
                    spawn_with_opts_at(&mut ctx, &tag, count, opts, pos_opts)?
                } else {
                    spawn_min_at(&mut ctx, &tag, count, pos_tag)?
                };
                Ok(Dynamic::from(spawned))
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
                ctx.set_patch(tag, patch, call_ctx.call_position())
            },
        );
        let ctx_for_unset = ctx.clone();
        engine.register_fn(
            "unset",
            move |call_ctx: NativeCallContext, tag: &str, path: &str| {
                let mut ctx = ctx_for_unset.lock().expect("lock script context");
                ctx.unset(tag, path, call_ctx.call_position())
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

        let ctx_for_remove_tag_str = ctx.clone();
        engine.register_fn("remove", move |call_ctx: NativeCallContext, tag: &str| {
            let mut ctx = ctx_for_remove_tag_str.lock().expect("lock script context");
            ctx.remove(tag, call_ctx.call_position())
        });

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
            spawn("lead", 1, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: 440.0 } });
            wait(1.0);
            scene("break");
            spawn("hit", 1, #{ body: #{ amp: 0.1 }, pitch: #{ center_hz: 880.0 } });
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
                    Action::Spawn { tag, .. } => {
                        if tag == "hit" {
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
                spawn("pad", 1, #{ body: #{ amp: 0.1 }, pitch: #{ center_hz: 200.0 } });
            });
            wait(0.2);
            spawn("after", 1, #{ body: #{ amp: 0.1 }, pitch: #{ center_hz: 300.0 } });
            end();
        "#,
        );

        let mut pad_time = None;
        let mut after_time = None;
        let mut finish_time = None;
        for ev in &scenario.events {
            for action in &ev.actions {
                match action {
                    Action::Spawn { tag, .. } => match tag.as_str() {
                        "pad" => pad_time = Some(ev.time),
                        "after" => after_time = Some(ev.time),
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
            spawn("tag", 3, #{ body: #{ amp: 0.25 }, pitch: #{ center_hz: 150.0 } });
            end();
        "#,
        );

        assert_eq!(scenario.scene_markers.len(), 1);
        assert_eq!(scenario.scene_markers[0].name, "alpha");
        assert_time_close(scenario.scene_markers[0].time, 0.0);

        let ev = scenario
            .events
            .iter()
            .find(|ev| ev.actions.iter().any(|a| matches!(a, Action::Spawn { .. })))
            .expect("spawn event");
        assert_time_close(ev.time, 1.2);
        assert_eq!(ev.actions.len(), 1);
        match &ev.actions[0] {
            Action::Spawn { count, tag, .. } => {
                assert_eq!(*count, 3);
                assert_eq!(tag, "tag");
            }
            other => panic!("unexpected action: {:?}", other),
        }
    }

    #[test]
    fn scene_created_when_scene_absent() {
        let scenario = run_script(
            r#"
            spawn("init", 1, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: 330.0 } });
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
                if let Action::Spawn { count, .. } = action {
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
            spawn("d", 1, #{ body: #{ amp: 0.2 }, pitch: #{ center_hz: 200.0 } });
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
            spawn("d", 1, #{ body: #{ ampp: 0.1 } });
        "#,
        );
        assert_ne!(pos, Position::NONE);
    }

    #[test]
    fn spawn_opts_bad_type_has_position() {
        let pos = eval_script_error_position(
            r#"
            spawn("d", 1, #{ body: #{ amp: "0.1" } });
        "#,
        );
        assert_ne!(pos, Position::NONE);
    }

    #[test]
    fn set_patch_unknown_key_has_position() {
        let pos = eval_script_error_position(
            r#"
            spawn("d", 1);
            set("d", #{ body: #{ amp: 0.1, ampp: 0.2 } });
        "#,
        );
        assert_ne!(pos, Position::NONE);
    }

    #[test]
    fn set_patch_bad_type_has_position() {
        let pos = eval_script_error_position(
            r#"
            spawn("d", 1);
            set("d", #{ body: #{ amp: "x" } });
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
                #{ body: #{ amm_BADKEY_123: 0.1 } }
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
                if let Action::Spawn { count, .. } = action {
                    assert!(*count > 0);
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
                if let Action::Set { patch, .. } = action {
                    let freq = patch
                        .as_object()
                        .and_then(|m| m.get("pitch"))
                        .and_then(|p| p.as_object())
                        .and_then(|m| m.get("center_hz"))
                        .and_then(|v| v.as_f64())
                        .map(|v| v as f32);
                    if let Some(freq) = freq {
                        freqs.push(freq);
                    }
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
