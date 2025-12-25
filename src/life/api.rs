use rhai::plugin::*;
use rhai::{EvalAltResult, FLOAT, Map, Module};

use super::scripting::ScriptContext;

#[export_module]
pub mod script_api {
    use super::*;

    /// Start a new named scene at the current cursor time.
    pub fn scene(ctx: &mut ScriptContext, name: &str) {
        ctx.scene(name);
    }

    /// Backward-compatible alias for `scene`.
    pub fn section(ctx: &mut ScriptContext, name: &str) {
        ctx.scene(name);
    }

    /// Advance the global time cursor by `sec` seconds.
    pub fn wait(ctx: &mut ScriptContext, sec: FLOAT) {
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
    /// `life_map` schema:
    /// - Legacy/entrain lifecycle: `{ type: "decay", initial_energy, half_life_sec, attack_sec? }`
    /// - Sustain lifecycle: `{ type: "sustain", initial_energy, metabolism_rate, recharge_rate?, action_cost?, envelope: { attack_sec, decay_sec, sustain_level } }`
    /// - Explicit brain: `{ brain: "seq", duration }` or `{ brain: "drone", sway? }`
    ///
    /// # Parameter Schemas
    ///
    /// ## Method Map (SpawnMethod)
    #[doc = include_str!("../../docs/schemas/spawn_method.md")]
    ///
    /// ## Life Map (BrainConfig)
    #[doc = include_str!("../../docs/schemas/brain_config.md")]
    #[rhai_fn(return_raw)]
    pub fn spawn(
        ctx: &mut ScriptContext,
        tag: &str,
        method_map: Map,
        life_map: Map,
        count: i64,
        amp: FLOAT,
    ) -> Result<(), Box<EvalAltResult>> {
        ctx.spawn(tag, method_map, life_map, count, amp as f32)
    }

    /// Alias for `spawn` to keep existing scripts working.
    #[rhai_fn(return_raw)]
    pub fn spawn_agents(
        ctx: &mut ScriptContext,
        tag: &str,
        method_map: Map,
        life_map: Map,
        count: i64,
        amp: FLOAT,
    ) -> Result<(), Box<EvalAltResult>> {
        ctx.spawn(tag, method_map, life_map, count, amp as f32)
    }

    /// Add a pure-tone agent at a fixed frequency.
    ///
    /// `life_map` schema: see `spawn` for the supported brain/lifecycle formats.
    #[rhai_fn(return_raw)]
    pub fn add_agent(
        ctx: &mut ScriptContext,
        tag: &str,
        freq: FLOAT,
        amp: FLOAT,
        life_map: Map,
    ) -> Result<(), Box<EvalAltResult>> {
        ctx.add_agent(tag, freq as f32, amp as f32, life_map)
    }

    /// Add an agent with an explicit synthesis kind.
    ///
    /// `kind` values:
    /// - `"pure_tone"`: `extra_map` may contain `phase`, `rhythm_freq`, `rhythm_sensitivity`,
    ///   `commitment`, `habituation_sensitivity` (all optional floats).
    /// - `"harmonic"`: `extra_map` includes the timbre genotype fields:
    ///   `mode`, `stiffness`, `brightness`, `comb`, `damping`, `vibrato_rate`, `vibrato_depth`,
    ///   `jitter`, `unison` plus the optional rhythm/commitment/habituation fields above.
    ///
    /// `life_map` schema: see `spawn` for the supported brain/lifecycle formats.
    ///
    /// # Parameter Schemas
    ///
    /// ## Extra Map (TimbreGenotype)
    #[doc = include_str!("../../docs/schemas/timbre_genotype.md")]
    ///
    /// ## Life Map (BrainConfig)
    #[doc = include_str!("../../docs/schemas/brain_config.md")]
    #[rhai_fn(name = "add_agent", return_raw)]
    pub fn add_agent_kind(
        ctx: &mut ScriptContext,
        tag: &str,
        kind: &str,
        freq: FLOAT,
        amp: FLOAT,
        extra_map: Map,
        life_map: Map,
    ) -> Result<(), Box<EvalAltResult>> {
        ctx.add_agent_kind(tag, kind, freq as f32, amp as f32, extra_map, life_map)
    }

    /// Set an agent's fundamental frequency in Hz.
    pub fn set_freq(ctx: &mut ScriptContext, target: &str, freq: FLOAT) {
        ctx.set_freq(target, freq as f32);
    }

    /// Set an agent's amplitude (linear gain).
    pub fn set_amp(ctx: &mut ScriptContext, target: &str, amp: FLOAT) {
        ctx.set_amp(target, amp as f32);
    }

    /// Set an agent's drift parameter.
    pub fn set_drift(ctx: &mut ScriptContext, target: &str, value: FLOAT) {
        ctx.set_drift(target, value as f32);
    }

    /// Set an agent's commitment parameter.
    pub fn set_commitment(ctx: &mut ScriptContext, target: &str, value: FLOAT) {
        ctx.set_commitment(target, value as f32);
    }

    /// Set an agent's habituation sensitivity.
    pub fn set_habituation_sensitivity(ctx: &mut ScriptContext, target: &str, value: FLOAT) {
        ctx.set_habituation_sensitivity(target, value as f32);
    }

    /// Set habituation parameters (weight, tau, max_depth).
    pub fn set_habituation_params(
        ctx: &mut ScriptContext,
        weight: FLOAT,
        tau: FLOAT,
        max_depth: FLOAT,
    ) {
        ctx.set_habituation_params(weight as f32, tau as f32, max_depth as f32);
    }

    /// Backward-compatible alias for `set_habituation_params` with max_depth = 1.0.
    pub fn set_habituation(ctx: &mut ScriptContext, weight: FLOAT, tau: FLOAT) {
        ctx.set_habituation_params(weight as f32, tau as f32, 1.0);
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

    /// Remove an agent (by tag or id) immediately.
    pub fn remove(ctx: &mut ScriptContext, target: &str) {
        ctx.remove(target);
    }

    /// Release an agent over `sec` seconds.
    pub fn release(ctx: &mut ScriptContext, target: &str, sec: FLOAT) {
        ctx.release(target, sec as f32);
    }

    /// Finish the scenario and stop playback.
    pub fn finish(ctx: &mut ScriptContext) {
        ctx.finish();
    }
}

pub fn module() -> Module {
    exported_module!(script_api)
}
