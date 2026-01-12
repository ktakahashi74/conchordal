use rhai::{EvalAltResult, FLOAT, Map, Position};

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

pub fn spawn_min_at(
    ctx: &mut ScriptContext,
    tag: &str,
    count: i64,
    position: Position,
) -> Result<i64, Box<EvalAltResult>> {
    ctx.spawn_default(tag, count, position)
}

pub fn spawn_with_third_at(
    ctx: &mut ScriptContext,
    tag: &str,
    count: i64,
    third: Map,
    position: Position,
) -> Result<i64, Box<EvalAltResult>> {
    ctx.spawn_with_third_map(tag, count, third, position)
}

pub fn spawn_with_opts_and_patch_at(
    ctx: &mut ScriptContext,
    tag: &str,
    count: i64,
    opts: Map,
    patch: Map,
    position: Position,
) -> Result<i64, Box<EvalAltResult>> {
    ctx.spawn_with_opts_and_patch(tag, count, Some(opts), Some(patch), position)
}
