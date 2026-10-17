use super::*;
use crate::core::landscape::Landscape;
use crate::core::timebase::Timebase;
use crate::life::community::Community;
use crate::life::voice::AnyArticulationCore;
use crate::life::voice::sound_body::SoundBody;
use crate::scenario::{
    DurationSpec, FieldSampling, FieldTarget, PhonationTiming, RhythmCouplingMode, RhythmRole,
};
use rand::SeedableRng;

fn run_script(src: &str) -> (Scenario, ScriptWarnings) {
    let ctx = Arc::new(Mutex::new(ScriptContext::default()));
    let engine = ScriptHost::create_engine(ctx.clone());
    let _ = engine.eval::<Dynamic>(src).expect("script runs");
    let mut ctx_out = ctx.lock().expect("lock script context");
    ctx_out.finish();
    (ctx_out.scenario.clone(), ctx_out.warnings.clone())
}

fn run_script_err(src: &str) -> ScriptError {
    let ctx = Arc::new(Mutex::new(ScriptContext::default()));
    let engine = ScriptHost::create_engine(ctx);
    let err = engine.eval::<Dynamic>(src).expect_err("script should fail");
    ScriptError::from_eval(err, None)
}

/// Like `run_script`, but seeds the context the way a CLI `--seed` override
/// does: `ScriptContext::with_seed` instead of a fresh random draw.
fn run_script_with_seed(src: &str, seed: u64) -> (Scenario, ScriptWarnings) {
    let ctx = Arc::new(Mutex::new(ScriptContext::with_seed(seed)));
    let engine = ScriptHost::create_engine(ctx.clone());
    let _ = engine.eval::<Dynamic>(src).expect("script runs");
    let mut ctx_out = ctx.lock().expect("lock script context");
    ctx_out.finish();
    (ctx_out.scenario.clone(), ctx_out.warnings.clone())
}

fn action_times(scenario: &Scenario) -> Vec<(f32, &Action)> {
    let mut out = Vec::new();
    for ev in &scenario.events {
        for action in &ev.actions {
            out.push((ev.time, action));
        }
    }
    out
}

fn first_spawn_spec_for_script(src: &str) -> VoiceSpec {
    let (scenario, _warnings) = run_script(src);
    scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.clone()),
            _ => None,
        })
        .expect("spawn action")
}

mod context;
mod placement;
mod spec;
