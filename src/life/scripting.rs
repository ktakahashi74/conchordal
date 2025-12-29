use std::collections::HashMap;
use std::fs;
use std::sync::{Arc, Mutex};

use anyhow::{Context, anyhow};
use rhai::{Engine, EvalAltResult, FLOAT, Map, Position};

use super::api::script_api as api;
use super::scenario::{Action, Event, IndividualConfig, LifeConfig, Scenario, Scene, SpawnMethod};

const SCRIPT_PRELUDE: &str = r#"
// Rhai-side helper to run blocks in parallel time branches
fn parallel(callback) {
    push_time();
    callback.call();
    pop_time();
}
"#;

#[derive(Debug, Clone)]
pub struct ScriptContext {
    pub cursor: f32,
    pub scene_start_time: f32,
    pub time_stack: Vec<f32>,
    pub scenario: Scenario,
    pub tag_counters: HashMap<String, usize>,
}

impl Default for ScriptContext {
    fn default() -> Self {
        Self {
            cursor: 0.0,
            scene_start_time: 0.0,
            time_stack: Vec::new(),
            scenario: Scenario { scenes: Vec::new() },
            tag_counters: HashMap::new(),
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
        let scene = self.current_scene_mut();
        scene.events.push(Event {
            time: rel_time,
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
    ) -> Result<(), Box<EvalAltResult>> {
        let method = Self::from_map::<SpawnMethod>(method_map, "SpawnMethod")?;
        let life = Self::from_map::<LifeConfig>(life_map, "LifeConfig")?;
        let c = count.max(0) as usize;
        let action = Action::SpawnAgents {
            method,
            count: c,
            amp,
            life,
            tag: Some(tag.to_string()),
        };
        self.push_event(vec![action]);
        Ok(())
    }

    pub fn add_agent(
        &mut self,
        tag: &str,
        freq: f32,
        amp: f32,
        life_map: Map,
    ) -> Result<(), Box<EvalAltResult>> {
        let life = Self::from_map::<LifeConfig>(life_map, "LifeConfig")?;
        let agent = IndividualConfig {
            freq,
            amp,
            life,
            tag: Some(tag.to_string()),
        };
        let action = Action::AddAgent { agent };
        self.push_event(vec![action]);
        Ok(())
    }

    pub fn set_freq(&mut self, target: &str, freq: f32) {
        self.push_event(vec![Action::SetFreq {
            target: target.to_string(),
            freq_hz: freq,
        }]);
    }

    pub fn set_amp(&mut self, target: &str, amp: f32) {
        self.push_event(vec![Action::SetAmp {
            target: target.to_string(),
            amp,
        }]);
    }

    pub fn set_drift(&mut self, target: &str, value: f32) {
        self.push_event(vec![Action::SetDrift {
            target: target.to_string(),
            value,
        }]);
    }

    pub fn set_commitment(&mut self, target: &str, value: f32) {
        self.push_event(vec![Action::SetCommitment {
            target: target.to_string(),
            value,
        }]);
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

    pub fn remove(&mut self, target: &str) {
        self.push_event(vec![Action::RemoveAgent {
            target: target.to_string(),
        }]);
    }

    pub fn release(&mut self, target: &str, sec: f32) {
        self.push_event(vec![Action::ReleaseAgent {
            target: target.to_string(),
            release_sec: sec,
        }]);
    }

    pub fn finish(&mut self) {
        self.push_event(vec![Action::Finish]);
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
                  -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_spawn_agents.lock().expect("lock script context");
                api::spawn_agents(&mut ctx, tag, method_map, life_map, count, amp)
            },
        );
        let ctx_for_add_agent = ctx.clone();
        engine.register_fn(
            "add_agent",
            move |tag: &str,
                  freq: FLOAT,
                  amp: FLOAT,
                  life_map: Map|
                  -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_add_agent.lock().expect("lock script context");
                api::add_agent(&mut ctx, tag, freq, amp, life_map)
            },
        );

        let ctx_for_set_freq = ctx.clone();
        engine.register_fn("set_freq", move |target: &str, freq: FLOAT| {
            let mut ctx = ctx_for_set_freq.lock().expect("lock script context");
            api::set_freq(&mut ctx, target, freq);
        });

        let ctx_for_set_amp = ctx.clone();
        engine.register_fn("set_amp", move |target: &str, amp: FLOAT| {
            let mut ctx = ctx_for_set_amp.lock().expect("lock script context");
            api::set_amp(&mut ctx, target, amp);
        });

        let ctx_for_set_drift = ctx.clone();
        engine.register_fn("set_drift", move |target: &str, value: FLOAT| {
            let mut ctx = ctx_for_set_drift.lock().expect("lock script context");
            api::set_drift(&mut ctx, target, value);
        });

        let ctx_for_set_commitment = ctx.clone();
        engine.register_fn("set_commitment", move |target: &str, value: FLOAT| {
            let mut ctx = ctx_for_set_commitment.lock().expect("lock script context");
            api::set_commitment(&mut ctx, target, value);
        });

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

        let ctx_for_remove = ctx.clone();
        engine.register_fn("remove", move |target: &str| {
            let mut ctx = ctx_for_remove.lock().expect("lock script context");
            api::remove(&mut ctx, target);
        });

        let ctx_for_release = ctx.clone();
        engine.register_fn("release", move |target: &str, sec: FLOAT| {
            let mut ctx = ctx_for_release.lock().expect("lock script context");
            api::release(&mut ctx, target, sec);
        });

        let ctx_for_finish = ctx.clone();
        engine.register_fn("finish", move || {
            let mut ctx = ctx_for_finish.lock().expect("lock script context");
            api::finish(&mut ctx);
        });

        engine
    }

    pub fn load_script(path: &str) -> anyhow::Result<Scenario> {
        let src = fs::read_to_string(path).with_context(|| format!("read script {path}"))?;
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

    fn run_script(src: &str) -> Scenario {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let script_src = format!("{SCRIPT_PRELUDE}\n{src}");
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
                temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5 },
                field: #{ core: "pitch_hill_climb" },
                modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            add_agent("lead", 440.0, 0.2, life);
            wait(1.0);
            scene("break");
            let hit_life = #{
                body: #{ core: "sine" },
                temporal: #{ core: "entrain", type: "decay", initial_energy: 0.8, half_life_sec: 0.2 },
                field: #{ core: "pitch_hill_climb" },
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
                    Action::AddAgent { agent } => {
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
                    temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.3 },
                    field: #{ core: "pitch_hill_climb" },
                    modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                    perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
                };
                add_agent("pad", 200.0, 0.1, life);
            });
            wait(0.2);
            let after_life = #{
                body: #{ core: "sine" },
                temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.3 },
                field: #{ core: "pitch_hill_climb" },
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
                    Action::AddAgent { agent } => match agent.tag.as_deref() {
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
                temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5 },
                field: #{ core: "pitch_hill_climb" },
                modulation: #{ core: "static", persistence: 0.5, exploration: 0.0 },
                perceptual: #{ tau_fast: 0.5, tau_slow: 6.0, w_boredom: 0.8, w_familiarity: 0.2 }
            };
            spawn_agents("tag", method, life, 3, 0.25);
        "#,
        );

        let ep = scenario.scenes.first().expect("scene exists");
        assert_eq!(ep.name.as_deref(), Some("alpha"));
        assert_time_close(ep.start_time, 0.0);
        assert_eq!(ep.events.len(), 1);

        let ev = &ep.events[0];
        assert_time_close(ev.time, 1.2);
        assert_eq!(ev.actions.len(), 1);
        match &ev.actions[0] {
            Action::SpawnAgents {
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
                    life.temporal,
                    crate::life::scenario::TemporalCoreConfig::Entrain { .. }
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
                temporal: #{ core: "entrain", type: "decay", initial_energy: 1.0, half_life_sec: 0.5 },
                field: #{ core: "pitch_hill_climb" },
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
