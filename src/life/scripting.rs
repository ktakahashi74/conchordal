use std::fs;
use std::sync::{Arc, Mutex};

use anyhow::{Context, anyhow};
use rhai::{Engine, EvalAltResult, FLOAT, Map, Position};

use super::api::script_api as api;
use super::scenario::{
    Action, AgentHandle, CohortHandle, Event, IndividualConfig, LifeConfig, Scenario, Scene,
    SpawnMethod, TagSelector, TargetRef,
};

const SCRIPT_PRELUDE: &str = r#"
// Rhai-side helper to run blocks in parallel time branches
fn parallel(callback) {
    push_time();
    callback.call();
    pop_time();
}
"#;

fn rewrite_minimal_spawn(src: &str) -> String {
    let out = src.replace("spawn (", "spawn_default(");
    out.replace("spawn(", "spawn_default(")
}

#[derive(Debug, Clone)]
pub struct ScriptContext {
    pub cursor: f32,
    pub scene_start_time: f32,
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
            scene_start_time: 0.0,
            time_stack: Vec::new(),
            scenario: Scenario { scenes: Vec::new() },
            next_group_id: 1,
            next_agent_id: 1,
            next_event_order: 1,
        }
    }
}

impl ScriptContext {
    fn current_scene_mut(&mut self) -> &mut Scene {
        if self.scenario.scenes.is_empty() {
            self.scenario.scenes.push(Scene {
                name: None,
                start_time: self.scene_start_time,
                events: Vec::new(),
            });
        }
        self.scenario.scenes.last_mut().expect("scene exists")
    }

    pub fn scene(&mut self, name: &str) {
        let scene = Scene {
            name: Some(name.to_string()),
            start_time: self.cursor,
            events: Vec::new(),
        };
        self.scene_start_time = self.cursor;
        self.scenario.scenes.push(scene);
    }

    pub fn wait(&mut self, sec: f32) {
        self.cursor += sec;
    }

    pub fn push_time(&mut self) {
        self.time_stack.push(self.cursor);
    }

    pub fn pop_time(&mut self) {
        if let Some(t) = self.time_stack.pop() {
            self.cursor = t;
        }
    }

    fn push_event(&mut self, actions: Vec<Action>) {
        let rel_time = self.cursor - self.scene_start_time;
        let order = self.next_event_order;
        self.next_event_order += 1;
        let scene = self.current_scene_mut();
        scene.events.push(Event {
            time: rel_time,
            order,
            repeat: None,
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
    ) -> Result<CohortHandle, Box<EvalAltResult>> {
        let method = Self::from_map::<SpawnMethod>(method_map, "SpawnMethod")?;
        let life = Self::from_map::<LifeConfig>(life_map, "LifeConfig")?;
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
        self.push_event(vec![action]);
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
        self.push_event(vec![action]);
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
    ) -> Result<AgentHandle, Box<EvalAltResult>> {
        let life = Self::from_map::<LifeConfig>(life_map, "LifeConfig")?;
        let agent = IndividualConfig {
            freq,
            amp,
            life,
            tag: Some(tag.to_string()),
        };
        let id = self.next_agent_id;
        self.next_agent_id += 1;
        let action = Action::AddAgent { id, agent };
        self.push_event(vec![action]);
        Ok(AgentHandle {
            id,
            tag: Some(tag.to_string()),
        })
    }

    pub fn set_freq(&mut self, target: TargetRef, freq: f32) {
        self.push_event(vec![Action::SetFreq {
            target,
            freq_hz: freq,
        }]);
    }

    pub fn set_amp(&mut self, target: TargetRef, amp: f32) {
        self.push_event(vec![Action::SetAmp { target, amp }]);
    }

    pub fn set_drift(&mut self, target: TargetRef, value: f32) {
        self.push_event(vec![Action::SetDrift { target, value }]);
    }

    pub fn set_commitment(&mut self, target: TargetRef, value: f32) {
        self.push_event(vec![Action::SetCommitment { target, value }]);
    }

    pub fn set_rhythm_vitality(&mut self, value: f32) {
        self.push_event(vec![Action::SetRhythmVitality { value }]);
    }

    pub fn set_global_coupling(&mut self, value: f32) {
        self.push_event(vec![Action::SetGlobalCoupling { value }]);
    }

    pub fn set_roughness_tolerance(&mut self, value: f32) {
        self.push_event(vec![Action::SetRoughnessTolerance { value }]);
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
        self.push_event(vec![Action::SetHarmonicity { mirror, limit }]);
        Ok(())
    }

    pub fn remove(&mut self, target: TargetRef) {
        self.push_event(vec![Action::RemoveAgent { target }]);
    }

    pub fn release(&mut self, target: TargetRef, sec: f32) {
        self.push_event(vec![Action::ReleaseAgent {
            target,
            release_sec: sec,
        }]);
    }

    pub fn finish(&mut self) {
        self.push_event(vec![Action::Finish]);
    }

    pub fn run(&mut self, sec: f32) {
        self.wait(sec.max(0.0));
        self.finish();
    }

    fn from_map<T: serde::de::DeserializeOwned>(
        map: Map,
        name: &str,
    ) -> Result<T, Box<EvalAltResult>> {
        serde_json::to_value(&map)
            .map_err(|e| {
                Box::new(EvalAltResult::ErrorRuntime(
                    format!("Error serializing {name}: {e}").into(),
                    Position::NONE,
                ))
            })
            .and_then(|v| {
                serde_json::from_value::<T>(v).map_err(|e| {
                    let debug_map = format!("{:?}", map);
                    Box::new(EvalAltResult::ErrorRuntime(
                        format!("Error parsing {name}: {e} (input: {debug_map})").into(),
                        Position::NONE,
                    ))
                })
            })
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
        engine.register_indexer_get(
            |cohort: &mut CohortHandle, index: i64| -> Result<AgentHandle, Box<EvalAltResult>> {
                if index < 0 {
                    return Err(Box::new(EvalAltResult::ErrorIndexNotFound(
                        index.into(),
                        Position::NONE,
                    )));
                }
                let idx = index as u32;
                if idx >= cohort.count {
                    return Err(Box::new(EvalAltResult::ErrorIndexNotFound(
                        index.into(),
                        Position::NONE,
                    )));
                }
                Ok(AgentHandle {
                    id: cohort.base_id + u64::from(idx),
                    tag: Some(cohort.tag.clone()),
                })
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

        let ctx_for_spawn_agents = ctx.clone();
        engine.register_fn(
            "spawn_agents",
            move |tag: &str,
                  method_map: Map,
                  life_map: Map,
                  count: i64,
                  amp: FLOAT|
                  -> Result<CohortHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_spawn_agents.lock().expect("lock script context");
                api::spawn_agents(&mut ctx, tag, method_map, life_map, count, amp)
            },
        );
        let ctx_for_spawn_default = ctx.clone();
        engine.register_fn(
            "spawn_default",
            move |tag: &str, count: i64| -> Result<CohortHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_spawn_default.lock().expect("lock script context");
                api::spawn_default(&mut ctx, tag, count)
            },
        );
        let ctx_for_add_agent = ctx.clone();
        engine.register_fn(
            "add_agent",
            move |tag: &str,
                  freq: FLOAT,
                  amp: FLOAT,
                  life_map: Map|
                  -> Result<AgentHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_add_agent.lock().expect("lock script context");
                api::add_agent(&mut ctx, tag, freq, amp, life_map)
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
            api::set_freq_agent(&mut ctx, target, freq);
        });
        let ctx_for_set_freq_cohort = ctx.clone();
        engine.register_fn("set_freq", move |target: CohortHandle, freq: FLOAT| {
            let mut ctx = ctx_for_set_freq_cohort.lock().expect("lock script context");
            api::set_freq_cohort(&mut ctx, target, freq);
        });
        let ctx_for_set_freq_tag = ctx.clone();
        engine.register_fn("set_freq", move |target: TagSelector, freq: FLOAT| {
            let mut ctx = ctx_for_set_freq_tag.lock().expect("lock script context");
            api::set_freq_tag(&mut ctx, target, freq);
        });

        let ctx_for_set_amp_agent = ctx.clone();
        engine.register_fn("set_amp", move |target: AgentHandle, amp: FLOAT| {
            let mut ctx = ctx_for_set_amp_agent.lock().expect("lock script context");
            api::set_amp_agent(&mut ctx, target, amp);
        });
        let ctx_for_set_amp_cohort = ctx.clone();
        engine.register_fn("set_amp", move |target: CohortHandle, amp: FLOAT| {
            let mut ctx = ctx_for_set_amp_cohort.lock().expect("lock script context");
            api::set_amp_cohort(&mut ctx, target, amp);
        });
        let ctx_for_set_amp_tag = ctx.clone();
        engine.register_fn("set_amp", move |target: TagSelector, amp: FLOAT| {
            let mut ctx = ctx_for_set_amp_tag.lock().expect("lock script context");
            api::set_amp_tag(&mut ctx, target, amp);
        });

        let ctx_for_set_drift_agent = ctx.clone();
        engine.register_fn("set_drift", move |target: AgentHandle, value: FLOAT| {
            let mut ctx = ctx_for_set_drift_agent.lock().expect("lock script context");
            api::set_drift_agent(&mut ctx, target, value);
        });
        let ctx_for_set_drift_cohort = ctx.clone();
        engine.register_fn("set_drift", move |target: CohortHandle, value: FLOAT| {
            let mut ctx = ctx_for_set_drift_cohort
                .lock()
                .expect("lock script context");
            api::set_drift_cohort(&mut ctx, target, value);
        });
        let ctx_for_set_drift_tag = ctx.clone();
        engine.register_fn("set_drift", move |target: TagSelector, value: FLOAT| {
            let mut ctx = ctx_for_set_drift_tag.lock().expect("lock script context");
            api::set_drift_tag(&mut ctx, target, value);
        });

        let ctx_for_set_commitment_agent = ctx.clone();
        engine.register_fn(
            "set_commitment",
            move |target: AgentHandle, value: FLOAT| {
                let mut ctx = ctx_for_set_commitment_agent
                    .lock()
                    .expect("lock script context");
                api::set_commitment_agent(&mut ctx, target, value);
            },
        );
        let ctx_for_set_commitment_cohort = ctx.clone();
        engine.register_fn(
            "set_commitment",
            move |target: CohortHandle, value: FLOAT| {
                let mut ctx = ctx_for_set_commitment_cohort
                    .lock()
                    .expect("lock script context");
                api::set_commitment_cohort(&mut ctx, target, value);
            },
        );
        let ctx_for_set_commitment_tag = ctx.clone();
        engine.register_fn(
            "set_commitment",
            move |target: TagSelector, value: FLOAT| {
                let mut ctx = ctx_for_set_commitment_tag
                    .lock()
                    .expect("lock script context");
                api::set_commitment_tag(&mut ctx, target, value);
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
        let script_src = format!("{SCRIPT_PRELUDE}\n{}", rewrite_minimal_spawn(&src));

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

    fn run_script(src: &str) -> Scenario {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = format!("{SCRIPT_PRELUDE}\n{}", rewrite_minimal_spawn(src));
        engine.eval::<()>(&script_src).expect("script runs");
        ctx.lock().expect("lock ctx").scenario.clone()
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
                modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("lead", 440.0, 0.2, life);
            wait(1.0);
            scene("break");
            let hit_life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 0.8, half_life_sec: 0.2 },
                pitch: #{ core: "pitch_hill_climb" },
                modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("hit", 880.0, 0.1, hit_life);
            wait(0.5);
            finish();
        "#,
        );

        assert_eq!(scenario.scenes.len(), 2);

        let intro = &scenario.scenes[0];
        assert_eq!(intro.name.as_deref(), Some("intro"));
        assert_time_close(intro.start_time, 0.0);
        assert_eq!(intro.events.len(), 1);
        assert_time_close(intro.events[0].time, 0.0);

        let break_ep = &scenario.scenes[1];
        assert_eq!(break_ep.name.as_deref(), Some("break"));
        assert_time_close(break_ep.start_time, 1.0);
        assert_eq!(break_ep.events.len(), 2);
        let mut has_hit = false;
        let mut has_finish = false;
        for ev in &break_ep.events {
            for action in &ev.actions {
                match action {
                    Action::AddAgent { agent, .. } => {
                        if agent.tag.as_deref() == Some("hit") {
                            assert_time_close(ev.time, 0.0);
                            has_hit = true;
                        }
                    }
                    Action::Finish => {
                        assert_time_close(ev.time, 0.5);
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
                    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                    perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
                };
                add_agent("pad", 200.0, 0.1, life);
            });
            wait(0.2);
            let after_life = #{
                body: #{ core: "sine" },
                articulation: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.3 },
                pitch: #{ core: "pitch_hill_climb" },
                modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("after", 300.0, 0.1, after_life);
            finish();
        "#,
        );

        let intro = &scenario.scenes.first().expect("intro scene should exist");
        assert_eq!(intro.events.len(), 3);

        let mut pad_time = None;
        let mut after_time = None;
        let mut finish_time = None;
        for ev in &intro.events {
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
                modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            spawn_agents("tag", method, life, 3, 0.25);
            finish();
        "#,
        );

        let ep = scenario.scenes.first().expect("scene exists");
        assert_eq!(ep.name.as_deref(), Some("alpha"));
        assert_time_close(ep.start_time, 0.0);
        assert_eq!(ep.events.len(), 2);

        let ev = &ep.events[0];
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
                modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("init", 330.0, 0.2, life);
            wait(0.3);
            finish();
        "#,
        );

        assert_eq!(scenario.scenes.len(), 1);
        let ep = &scenario.scenes[0];
        assert!(ep.name.is_none());
        assert_time_close(ep.start_time, 0.0);
        assert_eq!(ep.events.len(), 2);
        assert_time_close(ep.events[0].time, 0.0);
        assert_time_close(ep.events[1].time, 0.3);
    }

    #[test]
    fn sample_script_file_executes() {
        let scenario = ScriptHost::load_script("samples/01_fundamentals/spawn_basics.rhai")
            .expect("sample script should run");
        assert!(!scenario.scenes.is_empty());
    }

    #[test]
    fn minimal_spawn_run_executes() {
        let scenario = ScriptHost::load_script("samples/tests/minimal_spawn_run.rhai")
            .expect("minimal script should run");
        let scene = scenario.scenes.first().expect("scene exists");
        let mut has_spawn = false;
        for ev in &scene.events {
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
        let has_finish = scene
            .events
            .iter()
            .any(|ev| ev.actions.iter().any(|a| matches!(a, Action::Finish)));
        assert!(has_spawn, "expected spawn event");
        assert!(has_finish, "expected finish event");
    }

    #[test]
    fn empty_life_map_executes() {
        let scenario = ScriptHost::load_script("samples/tests/empty_life_map_ok.rhai")
            .expect("empty life map should run");
        let scene = scenario.scenes.first().expect("scene exists");
        let mut has_add = false;
        for ev in &scene.events {
            for action in &ev.actions {
                if let Action::AddAgent { id, .. } = action {
                    assert!(*id > 0);
                    has_add = true;
                }
            }
        }
        let has_finish = scene
            .events
            .iter()
            .any(|ev| ev.actions.iter().any(|a| matches!(a, Action::Finish)));
        assert!(has_add, "expected add_agent event");
        assert!(has_finish, "expected finish event");
    }

    #[test]
    fn handle_index_and_iter_executes() {
        let scenario = ScriptHost::load_script("samples/tests/handle_index_and_iter.rhai")
            .expect("handle iteration script should run");
        assert!(!scenario.scenes.is_empty());
    }

    #[test]
    fn tag_selector_ops_executes() {
        let scenario = ScriptHost::load_script("samples/tests/tag_selector_ops.rhai")
            .expect("tag selector script should run");
        assert!(!scenario.scenes.is_empty());
    }

    #[test]
    fn stable_order_same_time_is_monotonic() {
        let scenario = ScriptHost::load_script("samples/tests/stable_order_same_time.rhai")
            .expect("stable order script should run");
        let scene = scenario.scenes.first().expect("scene exists");
        let mut last_order = 0;
        let mut freqs = Vec::new();
        for ev in &scene.events {
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
