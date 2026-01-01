use rhai::plugin::*;
use rhai::{EvalAltResult, FLOAT, INT, Map, Module, Position};

use super::scenario::{AgentHandle, CohortHandle, TagSelector, TargetRef};
use super::scripting::ScriptContext;

#[export_module]
pub mod script_api {
    use super::*;

    /// Add a named scene marker at the current cursor time.
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

    /// Set the cursor to an absolute time (seconds).
    pub fn set_time(ctx: &mut ScriptContext, sec: FLOAT) {
        ctx.set_time(sec as f32);
    }

    // spawn/add_agent are registered explicitly in scripting with position-aware parsing.

    /// Create a dynamic selector for a tag.
    pub fn tag(_ctx: &mut ScriptContext, name: &str) -> TagSelector {
        TagSelector {
            tag: name.to_string(),
        }
    }

    /// Helper to build a spawn method map.
    fn method_map(mode: &str, min_freq: FLOAT, max_freq: FLOAT, opts: Option<Map>) -> Map {
        let mut map = Map::new();
        map.insert("mode".into(), mode.into());
        map.insert("min_freq".into(), min_freq.into());
        map.insert("max_freq".into(), max_freq.into());
        if let Some(opts) = opts {
            for (k, v) in opts {
                map.insert(k, v);
            }
        }
        map
    }

    /// Random log-uniform spawn method.
    #[rhai_fn(name = "random")]
    pub fn random_method(min_freq: FLOAT, max_freq: FLOAT) -> Map {
        method_map("random_log_uniform", min_freq, max_freq, None)
    }

    /// Random log-uniform spawn method with options.
    #[rhai_fn(name = "random")]
    pub fn random_method_opts(min_freq: FLOAT, max_freq: FLOAT, opts: Map) -> Map {
        method_map("random_log_uniform", min_freq, max_freq, Some(opts))
    }

    /// Harmonicity spawn method.
    pub fn harmonicity(min_freq: FLOAT, max_freq: FLOAT) -> Map {
        method_map("harmonicity", min_freq, max_freq, None)
    }

    /// Harmonicity spawn method with options.
    #[rhai_fn(name = "harmonicity")]
    pub fn harmonicity_opts(min_freq: FLOAT, max_freq: FLOAT, opts: Map) -> Map {
        method_map("harmonicity", min_freq, max_freq, Some(opts))
    }

    /// Spectral gap spawn method.
    pub fn spectral_gap(min_freq: FLOAT, max_freq: FLOAT) -> Map {
        method_map("spectral_gap", min_freq, max_freq, None)
    }

    /// Spectral gap spawn method with options.
    #[rhai_fn(name = "spectral_gap")]
    pub fn spectral_gap_opts(min_freq: FLOAT, max_freq: FLOAT, opts: Map) -> Map {
        method_map("spectral_gap", min_freq, max_freq, Some(opts))
    }

    /// Harmonic density spawn method.
    pub fn harmonic_density(min_freq: FLOAT, max_freq: FLOAT) -> Map {
        method_map("harmonic_density", min_freq, max_freq, None)
    }

    /// Harmonic density spawn method with options.
    #[rhai_fn(name = "harmonic_density")]
    pub fn harmonic_density_opts(min_freq: FLOAT, max_freq: FLOAT, opts: Map) -> Map {
        method_map("harmonic_density", min_freq, max_freq, Some(opts))
    }

    /// Life config patch helper (returns the patch map).
    pub fn life(patch: Map) -> Map {
        patch
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

    fn target_from_str(tag: &str) -> TargetRef {
        TargetRef::Tag {
            tag: tag.to_string(),
        }
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

    // set(...) variants are registered directly in scripting with position-aware paths.

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

    /// Remove agents by tag.
    #[rhai_fn(name = "remove")]
    pub fn remove_tag_str(ctx: &mut ScriptContext, tag: &str) {
        ctx.remove(target_from_str(tag));
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

    /// Release agents by tag over `sec` seconds.
    #[rhai_fn(name = "release")]
    pub fn release_tag_str(ctx: &mut ScriptContext, tag: &str, sec: FLOAT) {
        ctx.release(target_from_str(tag), sec as f32);
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

pub fn spawn_agents_at(
    ctx: &mut ScriptContext,
    tag: &str,
    method_map: Map,
    life_map: Map,
    count: i64,
    amp: FLOAT,
    position: Position,
) -> Result<CohortHandle, Box<EvalAltResult>> {
    ctx.spawn(tag, method_map, life_map, count, amp as f32, position)
}

pub fn spawn_min_at(
    ctx: &mut ScriptContext,
    tag: &str,
    count: i64,
    position: Position,
) -> Result<CohortHandle, Box<EvalAltResult>> {
    ctx.spawn_opts(tag, count, Map::new(), position)
}

pub fn spawn_with_opts_at(
    ctx: &mut ScriptContext,
    tag: &str,
    count: i64,
    opts: Map,
    position: Position,
) -> Result<CohortHandle, Box<EvalAltResult>> {
    ctx.spawn_opts(tag, count, opts, position)
}

pub fn add_agent_at(
    ctx: &mut ScriptContext,
    tag: &str,
    freq: FLOAT,
    amp: FLOAT,
    life_map: Map,
    position: Position,
) -> Result<AgentHandle, Box<EvalAltResult>> {
    ctx.add_agent(tag, freq as f32, amp as f32, life_map, position)
}

pub fn set_agent_at(
    ctx: &mut ScriptContext,
    target: AgentHandle,
    patch: Map,
    position: Position,
) -> Result<(), Box<EvalAltResult>> {
    ctx.set_patch(TargetRef::AgentId { id: target.id }, patch, position)
}

pub fn set_cohort_at(
    ctx: &mut ScriptContext,
    target: CohortHandle,
    patch: Map,
    position: Position,
) -> Result<(), Box<EvalAltResult>> {
    ctx.set_patch(
        TargetRef::Range {
            base_id: target.base_id,
            count: target.count,
        },
        patch,
        position,
    )
}

pub fn set_tag_at(
    ctx: &mut ScriptContext,
    target: TagSelector,
    patch: Map,
    position: Position,
) -> Result<(), Box<EvalAltResult>> {
    ctx.set_patch(TargetRef::Tag { tag: target.tag }, patch, position)
}

pub fn set_tag_str_at(
    ctx: &mut ScriptContext,
    tag: &str,
    patch: Map,
    position: Position,
) -> Result<(), Box<EvalAltResult>> {
    ctx.set_patch(
        TargetRef::Tag {
            tag: tag.to_string(),
        },
        patch,
        position,
    )
}

pub fn set_freq_agent(ctx: &mut ScriptContext, target: AgentHandle, freq: FLOAT) {
    let mut patch = Map::new();
    patch.insert("freq".into(), freq.into());
    let _ = ctx.set_patch(TargetRef::AgentId { id: target.id }, patch, Position::NONE);
}

pub fn set_freq_cohort(ctx: &mut ScriptContext, target: CohortHandle, freq: FLOAT) {
    let mut patch = Map::new();
    patch.insert("freq".into(), freq.into());
    let _ = ctx.set_patch(
        TargetRef::Range {
            base_id: target.base_id,
            count: target.count,
        },
        patch,
        Position::NONE,
    );
}

pub fn set_freq_tag(ctx: &mut ScriptContext, target: TagSelector, freq: FLOAT) {
    let mut patch = Map::new();
    patch.insert("freq".into(), freq.into());
    let _ = ctx.set_patch(TargetRef::Tag { tag: target.tag }, patch, Position::NONE);
}

pub fn set_amp_agent(ctx: &mut ScriptContext, target: AgentHandle, amp: FLOAT) {
    let mut patch = Map::new();
    patch.insert("amp".into(), amp.into());
    let _ = ctx.set_patch(TargetRef::AgentId { id: target.id }, patch, Position::NONE);
}

pub fn set_amp_cohort(ctx: &mut ScriptContext, target: CohortHandle, amp: FLOAT) {
    let mut patch = Map::new();
    patch.insert("amp".into(), amp.into());
    let _ = ctx.set_patch(
        TargetRef::Range {
            base_id: target.base_id,
            count: target.count,
        },
        patch,
        Position::NONE,
    );
}

pub fn set_amp_tag(ctx: &mut ScriptContext, target: TagSelector, amp: FLOAT) {
    let mut patch = Map::new();
    patch.insert("amp".into(), amp.into());
    let _ = ctx.set_patch(TargetRef::Tag { tag: target.tag }, patch, Position::NONE);
}

pub fn set_drift_agent(ctx: &mut ScriptContext, target: AgentHandle, drift: FLOAT) {
    let mut patch = Map::new();
    patch.insert("drift".into(), drift.into());
    let _ = ctx.set_patch(TargetRef::AgentId { id: target.id }, patch, Position::NONE);
}

pub fn set_drift_cohort(ctx: &mut ScriptContext, target: CohortHandle, drift: FLOAT) {
    let mut patch = Map::new();
    patch.insert("drift".into(), drift.into());
    let _ = ctx.set_patch(
        TargetRef::Range {
            base_id: target.base_id,
            count: target.count,
        },
        patch,
        Position::NONE,
    );
}

pub fn set_drift_tag(ctx: &mut ScriptContext, target: TagSelector, drift: FLOAT) {
    let mut patch = Map::new();
    patch.insert("drift".into(), drift.into());
    let _ = ctx.set_patch(TargetRef::Tag { tag: target.tag }, patch, Position::NONE);
}

pub fn set_commitment_agent(ctx: &mut ScriptContext, target: AgentHandle, commitment: FLOAT) {
    let mut patch = Map::new();
    patch.insert("commitment".into(), commitment.into());
    let _ = ctx.set_patch(TargetRef::AgentId { id: target.id }, patch, Position::NONE);
}

pub fn set_commitment_cohort(ctx: &mut ScriptContext, target: CohortHandle, commitment: FLOAT) {
    let mut patch = Map::new();
    patch.insert("commitment".into(), commitment.into());
    let _ = ctx.set_patch(
        TargetRef::Range {
            base_id: target.base_id,
            count: target.count,
        },
        patch,
        Position::NONE,
    );
}

pub fn set_commitment_tag(ctx: &mut ScriptContext, target: TagSelector, commitment: FLOAT) {
    let mut patch = Map::new();
    patch.insert("commitment".into(), commitment.into());
    let _ = ctx.set_patch(TargetRef::Tag { tag: target.tag }, patch, Position::NONE);
}

pub fn module() -> Module {
    exported_module!(script_api)
}
