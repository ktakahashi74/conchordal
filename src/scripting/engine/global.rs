#[cfg(feature = "native")]
use super::score;
use super::{
    Action, Arc, Array, BodyMethod, Bus, BusSet, ControlUpdateMode, Dynamic, Engine, EvalAltResult,
    FLOAT, FnPtr, INT, Mutex, NativeCallContext, PitchObjectiveMode, Placement, PopulationHandle,
    PopulationSpec, PopulationSpecHandle, ScaffoldConfig, ScenarioEnd, ScriptContext, parse_enum,
    sanitize01,
};

pub(super) fn register_scenario_builtins(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    #[cfg(feature = "native")]
    if let Some(stream) = ctx.lock().unwrap().stream.clone() {
        engine.on_progress(move |_| {
            use std::sync::atomic::Ordering;
            if stream.script_stop.load(Ordering::Acquire) {
                Some("stopped".into())
            } else if stream.bound_reached.load(Ordering::Acquire) {
                Some("score bound reached".into())
            } else {
                None
            }
        });
    }
    engine.disable_symbol("eval");
    engine.set_module_resolver(rhai::module_resolvers::DummyModuleResolver::new());
    let unavailable = |call: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
        Err(EvalAltResult::ErrorRuntime(
            "not available in scenarios; use wait() for score time".into(),
            call.call_position(),
        )
        .into())
    };
    engine.register_fn("timestamp", unavailable);
    engine.register_fn("sleep", move |call: NativeCallContext, _: FLOAT| {
        unavailable(call)
    });
    engine.register_fn("sleep", move |call: NativeCallContext, _: INT| {
        unavailable(call)
    });
    let ctx_endless = ctx.clone();
    engine.register_fn("endless", move |call: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
        let mut ctx = ctx_endless.lock().expect("lock script context");
        if ctx.parallel_used {
            return Err(EvalAltResult::ErrorRuntime(
                "parallel is not available in an endless scenario".into(), call.call_position(),
            ).into());
        }
        if ctx.scenario.end == ScenarioEnd::Open
            || ctx.cursor != 0.0
            || !ctx.scopes.is_empty()
            || ctx.emitted_any_event
            || call.call_level() > 1
        {
            return Err(EvalAltResult::ErrorRuntime(
                "endless must be declared once at top level before playback events and time advances".into(),
                call.call_position(),
            ).into());
        }
        ctx.scenario.end = ScenarioEnd::Open;
        #[cfg(feature = "native")]
        if let Some(stream) = ctx.stream.clone() {
            for marker in std::mem::take(&mut ctx.scenario.scene_markers) {
                stream.push(score::ScoreItem::Marker(marker)).map_err(|error|
                    Box::new(EvalAltResult::ErrorRuntime(error.message.into(), call.call_position())))?;
            }
        }
        Ok(())
    });
    engine.on_print(|msg| println!("[rhai] {msg}"));
}

pub(super) fn register_bus_constants(engine: &mut Engine) {
    let mut builtins = rhai::Module::new();
    builtins.set_var("habitat_bus", Bus::habitat());
    builtins.set_var("presentation_bus", Bus::presentation());
    engine.register_global_module(builtins.into());
}

pub(super) fn register_population_constructors(engine: &mut Engine) {
    engine.register_fn("sine", || PopulationSpecHandle {
        spec: PopulationSpec::preset(BodyMethod::Sine),
    });
    engine.register_fn("harmonic", || PopulationSpecHandle {
        spec: PopulationSpec::preset(BodyMethod::Harmonic),
    });
    engine.register_fn("modal", || PopulationSpecHandle {
        spec: PopulationSpec::preset(BodyMethod::Modal),
    });
    engine.register_fn("saw", || PopulationSpecHandle {
        spec: {
            let mut spec = PopulationSpec::preset(BodyMethod::Harmonic);
            spec.control.body.timbre.brightness = 0.85;
            spec
        },
    });
    engine.register_fn("square", || PopulationSpecHandle {
        spec: {
            let mut spec = PopulationSpec::preset(BodyMethod::Harmonic);
            spec.control.body.timbre.brightness = 0.65;
            spec
        },
    });
    engine.register_fn("noise", || PopulationSpecHandle {
        spec: {
            let mut spec = PopulationSpec::preset(BodyMethod::Harmonic);
            spec.control.body.timbre.brightness = 1.0;
            spec.control.body.timbre.motion = 1.0;
            spec
        },
    });
}

pub(super) fn register_bus_operators(engine: &mut Engine) {
    engine.register_fn("|", |left: Bus, right: Bus| left.set().combine(right.set()));
    engine.register_fn("|", |left: BusSet, right: Bus| left.combine(right.set()));
    engine.register_fn("|", |left: Bus, right: BusSet| left.set().combine(right));
    engine.register_fn("|", |left: BusSet, right: BusSet| left.combine(right));
}

pub(super) fn register_timeline(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    #[cfg(test)]
    {
        let ctx_for_create = ctx.clone();
        engine.register_fn(
            "create",
            move |call_ctx: NativeCallContext,
                  population_spec: PopulationSpecHandle,
                  count: INT| {
                let mut ctx = ctx_for_create.lock().expect("lock script context");
                ctx.create_population(population_spec, count, None, call_ctx.call_position())
            },
        );
    }
    let ctx_for_place_population_spec = ctx.clone();
    engine.register_fn(
        "place",
        move |call_ctx: NativeCallContext,
              population_spec: PopulationSpecHandle,
              placement: Placement|
              -> Result<PopulationHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_for_place_population_spec
                .lock()
                .expect("lock script context");
            ctx.place_population_spec(population_spec, placement, call_ctx.call_position())
        },
    );

    let ctx_for_wait = ctx.clone();
    engine.register_fn(
        "wait",
        move |_call_ctx: NativeCallContext, sec: FLOAT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_wait.lock().expect("lock script context");
            ctx.wait(sec, _call_ctx.call_position())
        },
    );
    let ctx_for_wait_int = ctx.clone();
    engine.register_fn(
        "wait",
        move |_call_ctx: NativeCallContext, sec: INT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_wait_int.lock().expect("lock script context");
            ctx.wait(sec as f64, _call_ctx.call_position())
        },
    );

    let ctx_for_flush = ctx.clone();
    engine.register_fn(
        "flush",
        move |_call_ctx: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_flush.lock().expect("lock script context");
            ctx.flush(_call_ctx.call_position());
            ctx.check_emission()
        },
    );

    #[cfg(feature = "research")]
    let ctx_for_felt = ctx.clone();
    engine.register_fn(
        "observe_felt_beat",
        move |call: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
            #[cfg(not(feature = "research"))]
            {
                Err(EvalAltResult::ErrorRuntime(
                    "observe_felt_beat requires the research feature".into(),
                    call.call_position(),
                )
                .into())
            }
            #[cfg(feature = "research")]
            {
                let mut ctx = ctx_for_felt.lock().expect("lock script context");
                ctx.check_header("observe_felt_beat", call.call_position())?;
                ctx.scenario.observe_felt_beat = true;
                ctx.felt_observation_position = Some(call.call_position());
                Ok(())
            }
        },
    );

    let ctx_for_tonal_observation = ctx.clone();
    engine.register_fn(
        "observe_tonal_context",
        move |call: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_tonal_observation
                .lock()
                .expect("lock script context");
            ctx.check_header("observe_tonal_context", call.call_position())?;
            ctx.scenario.observe_tonal_context = true;
            ctx.tonal_observation_position = Some(call.call_position());
            Ok(())
        },
    );

    let ctx_for_harmonic_observation = ctx.clone();
    engine.register_fn(
        "observe_harmonic_relations",
        move |call: NativeCallContext, max_sources: INT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_harmonic_observation.lock().expect("lock script context");
            let position = call.call_position();
            ctx.check_header("observe_harmonic_relations", position)?;
            if max_sources <= 0 || max_sources as u64 > crate::temporal_cognition::harmonic_relations::UMAX as u64 {
                return Err(EvalAltResult::ErrorRuntime(
                    "observe_harmonic_relations requires a positive source bound no larger than the structural-record capacity (38400)".into(), position).into());
            }
            ctx.scenario.harmonic_relation_sources = Some(max_sources as usize);
            ctx.harmonic_observation_position = Some(position);
            Ok(())
        },
    );

    let ctx_for_temporal_mode = ctx.clone();
    engine.register_fn(
        "temporal_mode",
        move |call: NativeCallContext, name: &str| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_temporal_mode.lock().expect("lock script context");
            ctx.check_header("temporal_mode", call.call_position())?;
            if ctx.temporal_mode_declared
                || ctx.cursor != 0.0
                || !ctx.scopes.is_empty()
                || ctx.emitted_any_event
                || call.call_level() > 1
            {
                return Err(EvalAltResult::ErrorRuntime(
                    "temporal_mode must be declared once at top level before playback events"
                        .into(),
                    call.call_position(),
                )
                .into());
            }
            let mode = parse_enum(
                "temporal_mode",
                name,
                call.call_position(),
                &[
                    ("off", crate::scenario::TemporalMode::Off),
                    ("observe", crate::scenario::TemporalMode::Observe),
                ],
            )?;
            #[cfg(not(feature = "research"))]
            if mode == crate::scenario::TemporalMode::Observe {
                return Err(EvalAltResult::ErrorRuntime(
                    "temporal_mode observe requires the research feature".into(),
                    call.call_position(),
                )
                .into());
            }
            ctx.scenario.temporal_mode = mode;
            ctx.temporal_mode_declared = true;
            Ok(())
        },
    );

    let ctx_room = ctx.clone();
    engine.register_fn(
        "room",
        move |call: NativeCallContext, mode: &str| -> Result<(), Box<EvalAltResult>> {
            use crate::config::RoomMode::*;
            let mut ctx = ctx_room.lock().expect("lock script context");
            let pos = call.call_position();
            ctx.check_header("room", pos)?;
            ctx.scenario.room = Some(parse_enum(
                "room",
                mode,
                pos,
                &[("off", Off), ("presentation", Presentation), ("both", Both)],
            )?);
            Ok(())
        },
    );

    macro_rules! register_output_gain {
        ($number:ty) => {
            let ctx_for_output_gain = ctx.clone();
            engine.register_fn(
                "output_gain",
                move |call: NativeCallContext, db: $number| -> Result<(), Box<EvalAltResult>> {
                    ctx_for_output_gain
                        .lock()
                        .expect("lock script context")
                        .set_output_gain(db as f64, call.call_position())
                },
            );
        };
    }
    register_output_gain!(FLOAT);
    register_output_gain!(INT);

    let ctx_for_seed = ctx.clone();
    engine.register_fn(
        "seed",
        move |call_ctx: NativeCallContext, seed: INT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_seed.lock().expect("lock script context");
            ctx.set_seed(seed, call_ctx.call_position())
        },
    );

    let ctx_for_release = ctx.clone();
    engine.register_fn(
        "release",
        move |_call_ctx: NativeCallContext, handle: PopulationHandle| {
            let mut ctx = ctx_for_release.lock().expect("lock script context");
            ctx.release_population(handle.id, _call_ctx.call_position());
        },
    );

    let ctx_for_profile = ctx.clone();
    engine.register_fn(
        "listener_profile",
        move |call: NativeCallContext, name: &str| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_profile.lock().expect("lock script context");
            let position = call.call_position();
            if ctx.cursor != 0.0
                || ctx.emitted_any_event
                || !ctx.scenario.scene_markers.is_empty()
                || !ctx.scopes.is_empty()
            {
                return Err(Box::new(EvalAltResult::ErrorRuntime(
                    "listener_profile must precede scene actions and scopes".into(),
                    position,
                )));
            }
            ctx.listener_profile = parse_enum(
                "listener_profile",
                name,
                position,
                &[("sensory", "sensory"), ("tonal", "tonal")],
            )?;
            Ok(())
        },
    );

    let ctx_for_section = ctx.clone();
    engine.register_fn(
        "section",
        move |call_ctx: NativeCallContext, name: &str, callback: FnPtr| {
            let start = {
                let mut ctx = ctx_for_section.lock().expect("lock script context");
                ctx.push_scene_marker(name, call_ctx.call_position());
                ctx.push_scope();
                ctx.cursor
            };
            let result = callback.call_within_context::<Dynamic>(&call_ctx, ());
            let mut ctx = ctx_for_section.lock().expect("lock script context");
            ctx.pop_scope(call_ctx.call_position());
            ctx.check_emission()?;
            if ctx.listener_profile == "tonal" {
                ctx.profile_section = Some(crate::scenario::expression::Expression {
                    start,
                    end: ctx.cursor,
                    arch: 1.0,
                    ritard: 1.0,
                    position: (0, 0),
                    from_profile: true,
                });
            }
            result.map(|_| ())
        },
    );

    let ctx_for_expression = ctx.clone();
    engine.register_fn(
        "expressive_section",
        move |call_ctx: NativeCallContext,
              name: &str,
              arch: Dynamic,
              ritard: Dynamic,
              callback: FnPtr| {
            let start = {
                let mut ctx = ctx_for_expression.lock().expect("lock script context");
                ctx.explicit_expression = true;
                ctx.push_scene_marker(name, call_ctx.call_position());
                ctx.push_scope();
                ctx.cursor
            };
            let result = callback.call_within_context::<Dynamic>(&call_ctx, ());
            let mut ctx = ctx_for_expression.lock().expect("lock script context");
            ctx.pop_scope(call_ctx.call_position());
            ctx.check_emission()?;
            let weight = |value: Dynamic| {
                value
                    .as_float()
                    .ok()
                    .or_else(|| value.as_int().ok().map(|n| n as f64))
            };
            let weights = weight(arch).zip(weight(ritard));
            let position = call_ctx.call_position();
            if let Some((arch, ritard)) =
                weights.filter(|(a, r)| a.is_finite() && r.is_finite() && *a >= 0.0 && *r >= 0.0)
            {
                if (arch != 0.0 || ritard != 0.0) && ctx.scenario.end == ScenarioEnd::Open {
                    ctx.warn(
                        "expression_open",
                        "expressive_section ignored: requires a finite final section".into(),
                        position,
                    );
                } else if arch != 0.0 || ritard != 0.0 {
                    let end = ctx.cursor;
                    ctx.expression_sections.push((
                        crate::scenario::expression::Expression {
                            start,
                            end,
                            from_profile: false,
                            arch: arch.min(1.0),
                            ritard: ritard.min(1.0),
                            position: (
                                position.line().unwrap_or(0),
                                position.position().unwrap_or(0),
                            ),
                        },
                        position,
                    ));
                }
            } else {
                ctx.warn(
                    "expression_weights",
                    "expressive_section ignored: weights must be finite and nonnegative".into(),
                    position,
                );
            }
            result.map(|_| ())
        },
    );

    let ctx_for_play = ctx.clone();
    engine.register_fn(
        "play",
        move |call_ctx: NativeCallContext, callback: FnPtr| {
            {
                let mut ctx = ctx_for_play.lock().expect("lock script context");
                ctx.push_scope();
            }
            let result = callback.call_within_context::<Dynamic>(&call_ctx, ());
            let mut ctx = ctx_for_play.lock().expect("lock script context");
            ctx.pop_scope(call_ctx.call_position());
            ctx.check_emission()?;
            result.map(|_| ())
        },
    );
    let ctx_for_play1 = ctx.clone();
    engine.register_fn(
        "play",
        move |call_ctx: NativeCallContext, callback: FnPtr, arg1: Dynamic| {
            {
                let mut ctx = ctx_for_play1.lock().expect("lock script context");
                ctx.push_scope();
            }
            let result = callback.call_within_context::<Dynamic>(&call_ctx, (arg1,));
            let mut ctx = ctx_for_play1.lock().expect("lock script context");
            ctx.pop_scope(call_ctx.call_position());
            ctx.check_emission()?;
            result.map(|_| ())
        },
    );
    let ctx_for_play2 = ctx.clone();
    engine.register_fn(
        "play",
        move |call_ctx: NativeCallContext, callback: FnPtr, arg1: Dynamic, arg2: Dynamic| {
            {
                let mut ctx = ctx_for_play2.lock().expect("lock script context");
                ctx.push_scope();
            }
            let result = callback.call_within_context::<Dynamic>(&call_ctx, (arg1, arg2));
            let mut ctx = ctx_for_play2.lock().expect("lock script context");
            ctx.pop_scope(call_ctx.call_position());
            ctx.check_emission()?;
            result.map(|_| ())
        },
    );
    let ctx_for_play3 = ctx.clone();
    engine.register_fn(
        "play",
        move |call_ctx: NativeCallContext,
              callback: FnPtr,
              arg1: Dynamic,
              arg2: Dynamic,
              arg3: Dynamic| {
            {
                let mut ctx = ctx_for_play3.lock().expect("lock script context");
                ctx.push_scope();
            }
            let result = callback.call_within_context::<Dynamic>(&call_ctx, (arg1, arg2, arg3));
            let mut ctx = ctx_for_play3.lock().expect("lock script context");
            ctx.pop_scope(call_ctx.call_position());
            ctx.check_emission()?;
            result.map(|_| ())
        },
    );
    let ctx_for_play_args = ctx.clone();
    engine.register_fn(
        "play",
        move |call_ctx: NativeCallContext, callback: FnPtr, args: Array| {
            {
                let mut ctx = ctx_for_play_args.lock().expect("lock script context");
                ctx.push_scope();
            }
            let result = callback.call_within_context::<Dynamic>(&call_ctx, args);
            let mut ctx = ctx_for_play_args.lock().expect("lock script context");
            ctx.pop_scope(call_ctx.call_position());
            ctx.check_emission()?;
            result.map(|_| ())
        },
    );

    let ctx_for_parallel = ctx.clone();
    engine.register_fn(
        "parallel",
        move |call_ctx: NativeCallContext, callbacks: Array| -> Result<(), Box<EvalAltResult>> {
            let start_time = {
                let mut ctx = ctx_for_parallel.lock().expect("lock script context");
                if ctx.scenario.end == ScenarioEnd::Open {
                    return Err(EvalAltResult::ErrorRuntime(
                        "parallel is not available in an endless scenario".into(),
                        call_ctx.call_position(),
                    )
                    .into());
                }
                ctx.parallel_used = true;
                ctx.cursor
            };
            let mut max_end = start_time;
            for (idx, callback) in callbacks.into_iter().enumerate() {
                let Some(fn_ptr) = callback.try_cast::<FnPtr>() else {
                    return Err(Box::new(EvalAltResult::ErrorRuntime(
                        format!("parallel expects closures (index {idx})").into(),
                        call_ctx.call_position(),
                    )));
                };
                {
                    let mut ctx = ctx_for_parallel.lock().expect("lock script context");
                    ctx.cursor = start_time;
                    ctx.push_scope();
                }
                let result = fn_ptr.call_within_context::<Dynamic>(&call_ctx, ());
                let mut ctx = ctx_for_parallel.lock().expect("lock script context");
                let end_time = ctx.cursor;
                // The next branch rewinds the cursor, so outer-handle actions left
                // pending here must be emitted at this branch's end.
                ctx.emit_pending_actions(call_ctx.call_position());
                ctx.pop_scope(call_ctx.call_position());
                max_end = max_end.max(end_time);
                let _ = result?;
            }
            let mut ctx = ctx_for_parallel.lock().expect("lock script context");
            ctx.cursor = max_end;
            Ok(())
        },
    );
}

pub(super) fn register_director(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    // Director-level shaping of the emergent production meter (scene-global
    // soft priors, symmetric to the consonance-field ops). These never
    // schedule a beat: `meter_stability` sets how readily a pulse forms,
    // `temporal_basin` sets the tempo region the pulse gravitates toward.
    let ctx_for_pitch_set = ctx.clone();
    let pitch_set_operation = move |call: NativeCallContext, value: f64| {
        let mut ctx = ctx_for_pitch_set.lock().expect("lock script context");
        let position = call.call_position();
        if ctx.cursor != 0.0
            || ctx.emitted_any_event
            || !ctx.scenario.scene_markers.is_empty()
            || !ctx.scopes.is_empty()
        {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "remember_pitch_set must precede scene actions and scopes".into(),
                position,
            )));
        }
        if !value.is_finite() || value < 0. {
            return Err(Box::new(EvalAltResult::ErrorRuntime(
                "remember_pitch_set requires a finite nonnegative weight".into(),
                position,
            )));
        }
        ctx.scenario.remember_pitch_set = value;
        ctx.scenario.pitch_set_position = position.line().zip(position.position());
        Ok::<_, Box<EvalAltResult>>(())
    };
    engine.register_fn("remember_pitch_set", pitch_set_operation.clone());
    engine.register_fn(
        "remember_pitch_set",
        move |call: NativeCallContext, value: INT| pitch_set_operation(call, value as f64),
    );
    let ctx_for_harmonic = ctx.clone();
    engine.register_fn(
        "harmonic_rhythm",
        move |call: NativeCallContext, value: INT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_harmonic.lock().expect("lock script context");
            ctx.check_header("harmonic_rhythm", call.call_position())?;
            if value < 0 {
                ctx.warn(
                    "harmonic_rhythm_value",
                    "harmonic_rhythm requires a nonnegative integer; previous setting retained"
                        .into(),
                    call.call_position(),
                );
            } else {
                ctx.scenario.harmonic_rhythm = value as u64;
                ctx.harmonic_position = call.call_position();
            }
            Ok(())
        },
    );
    let ctx_for_meter_stability_f = ctx.clone();
    engine.register_fn(
        "meter_stability",
        move |call: NativeCallContext, value: FLOAT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_meter_stability_f
                .lock()
                .expect("lock script context");
            ctx.check_header("meter_stability", call.call_position())?;
            let value = value as f32;
            if !value.is_finite() {
                ctx.warn(
                    "meter_stability",
                    format!(
                        "meter_stability() expects a finite value in [0, 1]; using {}",
                        sanitize01(value)
                    ),
                    call.call_position(),
                );
            }
            ctx.scenario.meter_shaping.stability = sanitize01(value);
            Ok(())
        },
    );
    let ctx_for_meter_stability_i = ctx.clone();
    engine.register_fn(
        "meter_stability",
        move |call: NativeCallContext, value: INT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_meter_stability_i
                .lock()
                .expect("lock script context");
            ctx.check_header("meter_stability", call.call_position())?;
            ctx.scenario.meter_shaping.stability = sanitize01(value as f32);
            Ok(())
        },
    );
    macro_rules! register_temporal_basin {
        ($min_ty:ty, $max_ty:ty) => {{
            let ctx_clone = ctx.clone();
            engine.register_fn(
                "temporal_basin",
                move |call: NativeCallContext,
                      min: $min_ty,
                      max: $max_ty|
                      -> Result<(), Box<EvalAltResult>> {
                    let mut ctx = ctx_clone.lock().expect("lock script context");
                    ctx.check_header("temporal_basin", call.call_position())?;
                    ctx.scenario.meter_shaping.basin_hz = Some((min as f32, max as f32));
                    Ok(())
                },
            );
        }};
    }
    register_temporal_basin!(FLOAT, FLOAT);
    register_temporal_basin!(INT, INT);
    register_temporal_basin!(FLOAT, INT);
    register_temporal_basin!(INT, FLOAT);
    let ctx_for_set_pitch_objective = ctx.clone();
    engine.register_fn(
        "set_pitch_objective",
        move |call: NativeCallContext, name: &str| -> Result<(), Box<EvalAltResult>> {
            let mode = parse_enum(
                "set_pitch_objective",
                name,
                call.call_position(),
                &[
                    ("consonance", PitchObjectiveMode::Consonance),
                    ("dissonance", PitchObjectiveMode::NegativeConsonance),
                ],
            )?;
            let mut ctx = ctx_for_set_pitch_objective
                .lock()
                .expect("lock script context");
            let cursor = ctx.cursor;
            let update = crate::core::landscape::LandscapeUpdate {
                pitch_objective_mode: Some(mode),
                ..Default::default()
            };
            ctx.push_event(
                cursor,
                vec![Action::UpdateLandscape { update }],
                call.call_position(),
            );
            Ok(())
        },
    );

    let ctx_for_set_global_coupling = ctx.clone();
    numeric_overloads!(engine, "set_global_coupling", |_call_ctx; value| {
            let mut ctx = ctx_for_set_global_coupling
                .lock()
                .expect("lock script context");
            let cursor = ctx.cursor;
            ctx.push_event(
                cursor,
                vec![Action::SetGlobalCoupling {
                    value: value as f32,
                }],
                _call_ctx.call_position(),
            );
        },
    );

    let ctx_for_set_control_update_mode = ctx.clone();
    engine.register_fn(
        "control_update_mode",
        move |call: NativeCallContext, name: &str| -> Result<(), Box<EvalAltResult>> {
            let mode = parse_enum(
                "control_update_mode",
                name,
                call.call_position(),
                &[
                    ("snapshot_phased", ControlUpdateMode::SnapshotPhased),
                    ("sequential_rotating", ControlUpdateMode::SequentialRotating),
                ],
            )?;
            let mut ctx = ctx_for_set_control_update_mode
                .lock()
                .expect("lock script context");
            ctx.check_header("control_update_mode", call.call_position())?;
            ctx.scenario.control_update_mode = mode;
            Ok(())
        },
    );

    let ctx_for_set_scaffold_off = ctx.clone();
    engine.register_fn(
        "scaffold_off",
        move |call: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_set_scaffold_off
                .lock()
                .expect("lock script context");
            ctx.check_header("scaffold_off", call.call_position())?;
            ctx.scenario.scaffold = ScaffoldConfig::Off;
            Ok(())
        },
    );
    let ctx_for_set_scaffold_shared = ctx.clone();
    engine.register_fn(
        "scaffold_shared",
        move |call: NativeCallContext, freq_hz: FLOAT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_set_scaffold_shared
                .lock()
                .expect("lock script context");
            ctx.check_header("scaffold_shared", call.call_position())?;
            ctx.scenario.scaffold = ScaffoldConfig::Shared {
                freq_hz: (freq_hz as f32).max(0.0),
            };
            Ok(())
        },
    );
    let ctx_for_set_scaffold_shared_int = ctx.clone();
    engine.register_fn(
        "scaffold_shared",
        move |call: NativeCallContext, freq_hz: INT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_set_scaffold_shared_int
                .lock()
                .expect("lock script context");
            ctx.check_header("scaffold_shared", call.call_position())?;
            ctx.scenario.scaffold = ScaffoldConfig::Shared {
                freq_hz: (freq_hz as f32).max(0.0),
            };
            Ok(())
        },
    );
    let ctx_for_set_scaffold_scrambled = ctx.clone();
    engine.register_fn(
        "scaffold_scrambled",
        move |call: NativeCallContext,
              freq_hz: FLOAT,
              seed: INT|
              -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_set_scaffold_scrambled
                .lock()
                .expect("lock script context");
            ctx.check_header("scaffold_scrambled", call.call_position())?;
            ctx.scenario.scaffold = ScaffoldConfig::Scrambled {
                freq_hz: (freq_hz as f32).max(0.0),
                seed: seed.max(0) as u64,
            };
            Ok(())
        },
    );
    let ctx_for_set_scaffold_scrambled_int = ctx.clone();
    engine.register_fn(
        "scaffold_scrambled",
        move |call: NativeCallContext, freq_hz: INT, seed: INT| -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_for_set_scaffold_scrambled_int
                .lock()
                .expect("lock script context");
            ctx.check_header("scaffold_scrambled", call.call_position())?;
            ctx.scenario.scaffold = ScaffoldConfig::Scrambled {
                freq_hz: (freq_hz as f32).max(0.0),
                seed: seed.max(0) as u64,
            };
            Ok(())
        },
    );

    let ctx_for_set_roughness_aversion = ctx.clone();
    numeric_overloads!(engine, "set_roughness_aversion", |_call_ctx; value| {
            let mut ctx = ctx_for_set_roughness_aversion
                .lock()
                .expect("lock script context");
            let cursor = ctx.cursor;
            ctx.push_event(
                cursor,
                vec![Action::SetRoughnessAversion {
                    value: value as f32,
                }],
                _call_ctx.call_position(),
            );
        },
    );

    let ctx_for_set_harmonicity_preference = ctx.clone();
    numeric_overloads!(engine, "set_harmonicity_preference", |_call_ctx; value| {
            let mut ctx = ctx_for_set_harmonicity_preference
                .lock()
                .expect("lock script context");
            let cursor = ctx.cursor;
            ctx.push_event(
                cursor,
                vec![Action::SetHarmonicityPreference {
                    value: value as f32,
                }],
                _call_ctx.call_position(),
            );
        },
    );

    for (continuation, name) in ["follow_harmonic_relations", "follow_harmonic_continuation"]
        .into_iter()
        .enumerate()
    {
        let ctx_harmonic = ctx.clone();
        let operation = move |call: NativeCallContext,
                              weight: FLOAT|
              -> Result<(), Box<EvalAltResult>> {
            let mut ctx = ctx_harmonic.lock().expect("lock script context");
            let position = call.call_position();
            if ctx.scenario.end == ScenarioEnd::Open
                && weight.is_finite()
                && weight != 0.
                && (-1. ..=1.).contains(&weight)
                && ctx.scenario.harmonic_relation_sources.is_none()
            {
                return Err(EvalAltResult::ErrorRuntime(
                    "declare observe_harmonic_relations(max_sources) before time advances to use a nonzero harmonic relation or continuation weight in an endless scenario".into(), position).into());
            }
            let action = if continuation == 0 {
                Action::FollowHarmonicRelations { weight }
            } else {
                Action::FollowHarmonicContinuation { weight }
            };
            let cursor = ctx.cursor;
            ctx.push_event(cursor, vec![action], position);
            ctx.check_emission()
        };
        engine.register_fn(name, operation.clone());
        engine.register_fn(name, move |call: NativeCallContext, weight: INT| {
            operation(call, weight as FLOAT)
        });
    }

    for (axis, name) in [
        "set_support_focus_preference",
        "set_support_conflict_preference",
        "set_related_novelty_preference",
    ]
    .into_iter()
    .enumerate()
    {
        let ctx_support = ctx.clone();
        let operation =
            move |call: NativeCallContext, value: FLOAT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_support.lock().expect("lock script context");
                let cursor = ctx.cursor;
                ctx.push_event(
                    cursor,
                    vec![Action::SetSupportPreference { axis, value }],
                    call.call_position(),
                );
                ctx.check_emission()
            };
        engine.register_fn(name, operation.clone());
        engine.register_fn(name, move |call: NativeCallContext, value: INT| {
            operation(call, value as FLOAT)
        });
    }
}
