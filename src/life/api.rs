use rhai::{EvalAltResult, FLOAT, Map, Position};

use super::scenario::{AgentHandle, CohortHandle, TagSelector, TargetRef};
use super::scripting::ScriptContext;

pub fn push_time(ctx: &mut ScriptContext, position: Position) -> Result<(), Box<EvalAltResult>> {
    ctx.push_time(position)
}

pub fn pop_time(ctx: &mut ScriptContext, position: Position) -> Result<(), Box<EvalAltResult>> {
    ctx.pop_time(position)
}

pub fn set_time(
    ctx: &mut ScriptContext,
    sec: FLOAT,
    position: Position,
) -> Result<(), Box<EvalAltResult>> {
    ctx.set_time(sec as f32, position)
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

pub fn set_plan_rate_source(ctx: &mut ScriptContext, source_id: i64, rate: FLOAT) {
    ctx.set_plan_rate_source(source_id, rate as f32);
}

pub fn set_plan_rate_tag(ctx: &mut ScriptContext, tag: &str, rate: FLOAT) {
    ctx.set_plan_rate_tag(tag, rate as f32);
}
