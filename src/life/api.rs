use rhai::plugin::*;
use rhai::{EvalAltResult, FLOAT, INT, Map, Module};

use super::scenario::{AgentHandle, CohortHandle, TagSelector, TargetRef};
use super::scripting::ScriptContext;

#[export_module]
pub mod script_api {
    use super::*;

    /// Start a new named scene at the current cursor time.
    pub fn scene(ctx: &mut ScriptContext, name: &str) {
        ctx.scene(name);
    }

    /// Advance the global time cursor by `sec` seconds.
    pub fn wait(ctx: &mut ScriptContext, sec: FLOAT) {
        ctx.wait(sec as f32);
    }

    /// Advance the global time cursor by `sec` seconds.
    #[rhai_fn(name = "wait")]
    pub fn wait_int(ctx: &mut ScriptContext, sec: INT) {
        ctx.wait(sec as f32);
    }

    /// Save the current cursor time onto the time stack.
    pub fn push_time(ctx: &mut ScriptContext) {
        ctx.push_time();
    }

    /// Restore the most recently pushed cursor time.
    pub fn pop_time(ctx: &mut ScriptContext) {
        ctx.pop_time();
    }

    /// Spawn multiple agents using a frequency selection method and life config.
    ///
    /// `method_map` schema:
    /// - `mode`: `"harmonicity" | "low_harmonicity" | "harmonic_density" | "zero_crossing" | "spectral_gap" | "random_log_uniform"`
    /// - `min_freq`: f32 (Hz)
    /// - `max_freq`: f32 (Hz)
    /// - `min_dist_erb`: f32 (optional, minimum ERB spacing)
    /// - `temperature`: f32 (optional, only for `harmonic_density`)
    ///
    /// `life_map` schema: see `life_config` for required body/articulation/pitch/modulation cores.
    ///
    /// # Parameter Schemas
    ///
    /// ## Method Map (SpawnMethod)
    #[doc = include_str!("../../docs/schemas/spawn_method.md")]
    ///
    /// ## Life Map (LifeConfig)
    #[doc = include_str!("../../docs/schemas/life_config.md")]
    #[rhai_fn(return_raw)]
    pub fn spawn_agents(
        ctx: &mut ScriptContext,
        tag: &str,
        method_map: Map,
        life_map: Map,
        count: i64,
        amp: FLOAT,
    ) -> Result<CohortHandle, Box<EvalAltResult>> {
        ctx.spawn(tag, method_map, life_map, count, amp as f32)
    }

    /// Spawn agents with default method/life settings.
    #[rhai_fn(name = "spawn_default", return_raw)]
    pub fn spawn_default(
        ctx: &mut ScriptContext,
        tag: &str,
        count: i64,
    ) -> Result<CohortHandle, Box<EvalAltResult>> {
        ctx.spawn_default(tag, count)
    }

    /// Add an agent at a fixed frequency.
    ///
    /// `life_map` schema: see `spawn` for the required body/core configs.
    #[rhai_fn(return_raw)]
    pub fn add_agent(
        ctx: &mut ScriptContext,
        tag: &str,
        freq: FLOAT,
        amp: FLOAT,
        life_map: Map,
    ) -> Result<AgentHandle, Box<EvalAltResult>> {
        ctx.add_agent(tag, freq as f32, amp as f32, life_map)
    }

    /// Create a dynamic selector for a tag.
    pub fn tag(_ctx: &mut ScriptContext, name: &str) -> TagSelector {
        TagSelector {
            tag: name.to_string(),
        }
    }

    fn target_from_agent(agent: AgentHandle) -> TargetRef {
        TargetRef::AgentId { id: agent.id }
    }

    fn target_from_cohort(cohort: CohortHandle) -> TargetRef {
        TargetRef::Range {
            base_id: cohort.base_id,
            count: cohort.count,
        }
    }

    fn target_from_tag(selector: TagSelector) -> TargetRef {
        TargetRef::Tag { tag: selector.tag }
    }

    /// Set an agent's fundamental frequency in Hz.
    pub fn set_freq_agent(ctx: &mut ScriptContext, target: AgentHandle, freq: FLOAT) {
        ctx.set_freq(target_from_agent(target), freq as f32);
    }

    /// Set a cohort's fundamental frequency in Hz.
    pub fn set_freq_cohort(ctx: &mut ScriptContext, target: CohortHandle, freq: FLOAT) {
        ctx.set_freq(target_from_cohort(target), freq as f32);
    }

    /// Set a tag selector's fundamental frequency in Hz.
    pub fn set_freq_tag(ctx: &mut ScriptContext, target: TagSelector, freq: FLOAT) {
        ctx.set_freq(target_from_tag(target), freq as f32);
    }

    /// Set an agent's amplitude (linear gain).
    pub fn set_amp_agent(ctx: &mut ScriptContext, target: AgentHandle, amp: FLOAT) {
        ctx.set_amp(target_from_agent(target), amp as f32);
    }

    /// Set a cohort's amplitude (linear gain).
    pub fn set_amp_cohort(ctx: &mut ScriptContext, target: CohortHandle, amp: FLOAT) {
        ctx.set_amp(target_from_cohort(target), amp as f32);
    }

    /// Set a tag selector's amplitude (linear gain).
    pub fn set_amp_tag(ctx: &mut ScriptContext, target: TagSelector, amp: FLOAT) {
        ctx.set_amp(target_from_tag(target), amp as f32);
    }

    /// Set an agent's drift parameter.
    pub fn set_drift_agent(ctx: &mut ScriptContext, target: AgentHandle, value: FLOAT) {
        ctx.set_drift(target_from_agent(target), value as f32);
    }

    /// Set a cohort's drift parameter.
    pub fn set_drift_cohort(ctx: &mut ScriptContext, target: CohortHandle, value: FLOAT) {
        ctx.set_drift(target_from_cohort(target), value as f32);
    }

    /// Set a tag selector's drift parameter.
    pub fn set_drift_tag(ctx: &mut ScriptContext, target: TagSelector, value: FLOAT) {
        ctx.set_drift(target_from_tag(target), value as f32);
    }

    /// Set an agent's commitment parameter.
    pub fn set_commitment_agent(ctx: &mut ScriptContext, target: AgentHandle, value: FLOAT) {
        ctx.set_commitment(target_from_agent(target), value as f32);
    }

    /// Set a cohort's commitment parameter.
    pub fn set_commitment_cohort(ctx: &mut ScriptContext, target: CohortHandle, value: FLOAT) {
        ctx.set_commitment(target_from_cohort(target), value as f32);
    }

    /// Set a tag selector's commitment parameter.
    pub fn set_commitment_tag(ctx: &mut ScriptContext, target: TagSelector, value: FLOAT) {
        ctx.set_commitment(target_from_tag(target), value as f32);
    }

    /// Set the global rhythm vitality (affects oscillatory dynamics).
    pub fn set_rhythm_vitality(ctx: &mut ScriptContext, value: FLOAT) {
        ctx.set_rhythm_vitality(value as f32);
    }

    /// Set the global coupling strength across agents.
    pub fn set_global_coupling(ctx: &mut ScriptContext, value: FLOAT) {
        ctx.set_global_coupling(value as f32);
    }

    /// Set the global roughness tolerance.
    pub fn set_roughness_tolerance(ctx: &mut ScriptContext, value: FLOAT) {
        ctx.set_roughness_tolerance(value as f32);
    }

    /// Configure harmonicity calculation parameters.
    ///
    /// `map` schema:
    /// - `mirror`: f32 (optional, mirror weighting)
    /// - `limit`: i64 (optional, partials limit)
    #[rhai_fn(return_raw)]
    pub fn set_harmonicity(ctx: &mut ScriptContext, map: Map) -> Result<(), Box<EvalAltResult>> {
        ctx.set_harmonicity(map)
    }

    /// Remove agents immediately.
    pub fn remove_agent(ctx: &mut ScriptContext, target: AgentHandle) {
        ctx.remove(target_from_agent(target));
    }

    /// Remove a cohort immediately.
    pub fn remove_cohort(ctx: &mut ScriptContext, target: CohortHandle) {
        ctx.remove(target_from_cohort(target));
    }

    /// Remove a tag selector immediately.
    pub fn remove_tag(ctx: &mut ScriptContext, target: TagSelector) {
        ctx.remove(target_from_tag(target));
    }

    /// Release agents over `sec` seconds.
    pub fn release_agent(ctx: &mut ScriptContext, target: AgentHandle, sec: FLOAT) {
        ctx.release(target_from_agent(target), sec as f32);
    }

    /// Release a cohort over `sec` seconds.
    pub fn release_cohort(ctx: &mut ScriptContext, target: CohortHandle, sec: FLOAT) {
        ctx.release(target_from_cohort(target), sec as f32);
    }

    /// Release a tag selector over `sec` seconds.
    pub fn release_tag(ctx: &mut ScriptContext, target: TagSelector, sec: FLOAT) {
        ctx.release(target_from_tag(target), sec as f32);
    }

    /// Finish the scenario and stop playback.
    pub fn finish(ctx: &mut ScriptContext) {
        ctx.finish();
    }

    /// Run for `sec` seconds and finish the scenario.
    #[rhai_fn(name = "run")]
    pub fn run_float(ctx: &mut ScriptContext, sec: FLOAT) {
        ctx.run(sec as f32);
    }

    /// Run for `sec` seconds and finish the scenario.
    #[rhai_fn(name = "run")]
    pub fn run_int(ctx: &mut ScriptContext, sec: INT) {
        ctx.run(sec as f32);
    }
}

pub fn module() -> Module {
    exported_module!(script_api)
}
