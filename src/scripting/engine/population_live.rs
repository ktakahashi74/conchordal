use super::population::PopulationSpecNumericSetter;
use super::{
    Arc, ControlUpdate, Engine, EvalAltResult, FLOAT, INT, Mutex, NativeCallContext,
    PitchApplyMode, PopulationHandle, PopulationSpec, PopulationSpecHandle, PopulationStatus,
    Position, ScenarioEnd, ScriptContext, ScriptWarning,
};

pub(super) fn register_live_methods(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    register_population_numeric_methods(
        engine,
        ctx.clone(),
        &[
            ("amp", PopulationSpec::set_amp, patch_amp),
            ("freq", PopulationSpec::set_freq, patch_freq),
            (
                "sustain_drive",
                PopulationSpec::set_continuous_drive,
                patch_continuous_drive,
            ),
        ],
    );
    register_population_crowding_overloads(engine, ctx.clone(), "avoid_neighbors");
    let ctx_for_population_leave_self_out = ctx.clone();
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "leave_self_out",
        move |call_ctx: NativeCallContext,
              handle: PopulationHandle,
              enabled: bool|
              -> Result<PopulationHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_for_population_leave_self_out
                .lock()
                .expect("lock script context");
            let endless = ctx.scenario.end == ScenarioEnd::Open;
            let Some(population) = ctx.populations.get_mut(&handle.id) else {
                diagnostics
                    .lock()
                    .expect("lock script warnings")
                    .push(ScriptWarning {
                        kind: "leave_self_out_absent",
                        count: 1,
                        message: format!(
                            "leave_self_out ignored for {} population {}",
                            ctx.population_absence(handle.id),
                            handle.id
                        ),
                        position: call_ctx.call_position(),
                    });
                return Ok(handle);
            };
            match population.status {
                PopulationStatus::Live => {
                    if endless && enabled && !population.loo_at_placement {
                        return Err(EvalAltResult::ErrorRuntime(
                            "enable leave_self_out at placement in an endless scenario".into(),
                            call_ctx.call_position(),
                        )
                        .into());
                    }
                    population.spec.set_leave_self_out(enabled);
                    population.pending_patch.leave_self_out = Some(enabled);
                }
                _ => ctx.warn_population_inactive(
                    handle.id,
                    "leave_self_out",
                    call_ctx.call_position(),
                ),
            }
            Ok(handle)
        },
    );
    register_population_numeric_methods(
        engine,
        ctx.clone(),
        &[
            (
                "move_cost",
                PopulationSpec::set_move_cost_coeff,
                patch_move_cost_coeff,
            ),
            (
                "proposal_interval",
                PopulationSpec::set_proposal_interval_sec,
                patch_proposal_interval,
            ),
        ],
    );
    let ctx_for_population_global_peaks = ctx.clone();
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "global_peaks",
        move |call_ctx: NativeCallContext,
              handle: PopulationHandle,
              count: INT|
              -> Result<PopulationHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_for_population_global_peaks
                .lock()
                .expect("lock script context");
            let Some(population) = ctx.populations.get_mut(&handle.id) else {
                diagnostics
                    .lock()
                    .expect("lock script warnings")
                    .push(ScriptWarning {
                        kind: "global_peaks_absent",
                        count: 1,
                        message: format!(
                            "global_peaks ignored for {} population {}",
                            ctx.population_absence(handle.id),
                            handle.id
                        ),
                        position: call_ctx.call_position(),
                    });
                return Ok(handle);
            };
            match population.status {
                PopulationStatus::Live => {
                    population.spec.set_global_peaks(count, 0.0);
                    population.pending_patch.global_peak_count = Some(count);
                    population.pending_patch.global_peak_min_sep_cents = Some(0.0);
                }
                _ => ctx.warn_population_inactive(
                    handle.id,
                    "global_peaks",
                    call_ctx.call_position(),
                ),
            }
            Ok(handle)
        },
    );
    let ctx_for_population_global_peaks_sep = ctx.clone();
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "global_peaks",
        move |call_ctx: NativeCallContext,
              handle: PopulationHandle,
              count: INT,
              min_sep_cents: FLOAT|
              -> Result<PopulationHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_for_population_global_peaks_sep
                .lock()
                .expect("lock script context");
            let Some(population) = ctx.populations.get_mut(&handle.id) else {
                diagnostics
                    .lock()
                    .expect("lock script warnings")
                    .push(ScriptWarning {
                        kind: "global_peaks_absent",
                        count: 1,
                        message: format!(
                            "global_peaks ignored for {} population {}",
                            ctx.population_absence(handle.id),
                            handle.id
                        ),
                        position: call_ctx.call_position(),
                    });
                return Ok(handle);
            };
            let min_sep = min_sep_cents as f32;
            match population.status {
                PopulationStatus::Live => {
                    population.spec.set_global_peaks(count, min_sep);
                    population.pending_patch.global_peak_count = Some(count);
                    population.pending_patch.global_peak_min_sep_cents = Some(min_sep);
                }
                _ => ctx.warn_population_inactive(
                    handle.id,
                    "global_peaks",
                    call_ctx.call_position(),
                ),
            }
            Ok(handle)
        },
    );
    let ctx_for_population_global_peaks_sep_int = ctx.clone();
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "global_peaks",
        move |call_ctx: NativeCallContext,
              handle: PopulationHandle,
              count: INT,
              min_sep_cents: INT|
              -> Result<PopulationHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_for_population_global_peaks_sep_int
                .lock()
                .expect("lock script context");
            let Some(population) = ctx.populations.get_mut(&handle.id) else {
                diagnostics
                    .lock()
                    .expect("lock script warnings")
                    .push(ScriptWarning {
                        kind: "global_peaks_absent",
                        count: 1,
                        message: format!(
                            "global_peaks ignored for {} population {}",
                            ctx.population_absence(handle.id),
                            handle.id
                        ),
                        position: call_ctx.call_position(),
                    });
                return Ok(handle);
            };
            let min_sep = min_sep_cents as f32;
            match population.status {
                PopulationStatus::Live => {
                    population.spec.set_global_peaks(count, min_sep);
                    population.pending_patch.global_peak_count = Some(count);
                    population.pending_patch.global_peak_min_sep_cents = Some(min_sep);
                }
                _ => ctx.warn_population_inactive(
                    handle.id,
                    "global_peaks",
                    call_ctx.call_position(),
                ),
            }
            Ok(handle)
        },
    );
    let ctx_for_population_ratio_candidates = ctx.clone();
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "ratio_candidates",
        move |call_ctx: NativeCallContext,
              handle: PopulationHandle,
              count: INT|
              -> Result<PopulationHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_for_population_ratio_candidates
                .lock()
                .expect("lock script context");
            let Some(population) = ctx.populations.get_mut(&handle.id) else {
                diagnostics
                    .lock()
                    .expect("lock script warnings")
                    .push(ScriptWarning {
                        kind: "ratio_candidates_absent",
                        count: 1,
                        message: format!(
                            "ratio_candidates ignored for {} population {}",
                            ctx.population_absence(handle.id),
                            handle.id
                        ),
                        position: call_ctx.call_position(),
                    });
                return Ok(handle);
            };
            match population.status {
                PopulationStatus::Live => {
                    population.spec.set_ratio_candidates(count);
                    population.pending_patch.ratio_candidate_count = Some(count);
                    population.pending_patch.use_ratio_candidates = Some(count > 0);
                }
                _ => ctx.warn_population_inactive(
                    handle.id,
                    "ratio_candidates",
                    call_ctx.call_position(),
                ),
            }
            Ok(handle)
        },
    );
    register_population_numeric_methods(
        engine,
        ctx.clone(),
        &[(
            "temperature",
            PopulationSpec::set_temperature,
            patch_temperature,
        )],
    );
    let ctx_for_population_move_cost_time_scale = ctx.clone();
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "move_cost_time_scale",
        move |call: NativeCallContext,
              handle: PopulationHandle,
              name: &str|
              -> Result<PopulationHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_for_population_move_cost_time_scale
                .lock()
                .expect("lock script context");
            let Some(population) = ctx.populations.get_mut(&handle.id) else {
                diagnostics
                    .lock()
                    .expect("lock script warnings")
                    .push(ScriptWarning {
                        kind: "move_cost_time_scale_absent",
                        count: 1,
                        message: format!(
                            "move_cost_time_scale ignored for {} population {}",
                            ctx.population_absence(handle.id),
                            handle.id
                        ),
                        position: call.call_position(),
                    });
                return Ok(handle);
            };
            population
                .spec
                .set_move_cost_time_scale(name, call.call_position())?;
            let value = population.spec.control.pitch.move_cost_time_scale;
            match population.status {
                PopulationStatus::Live => {
                    population
                        .spec
                        .control
                        .pitch
                        .set_move_cost_time_scale(value);
                    population.pending_patch.move_cost_time_scale = Some(value);
                }
                _ => ctx.warn_population_inactive(
                    handle.id,
                    "move_cost_time_scale",
                    call.call_position(),
                ),
            }
            Ok(handle)
        },
    );
    let ctx_for_population_pitch_apply_mode = ctx.clone();
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "pitch_apply_mode",
        move |call: NativeCallContext,
              handle: PopulationHandle,
              name: &str|
              -> Result<PopulationHandle, Box<EvalAltResult>> {
            let mut ctx = ctx_for_population_pitch_apply_mode
                .lock()
                .expect("lock script context");
            ctx.record_harmonic_site(handle.id, "pitch_apply_mode", call.call_position());
            let Some(population) = ctx.populations.get_mut(&handle.id) else {
                diagnostics
                    .lock()
                    .expect("lock script warnings")
                    .push(ScriptWarning {
                        kind: "pitch_apply_mode_absent",
                        count: 1,
                        message: format!(
                            "pitch_apply_mode ignored for {} population {}",
                            ctx.population_absence(handle.id),
                            handle.id
                        ),
                        position: call.call_position(),
                    });
                return Ok(handle);
            };
            let mut next_spec = population.spec.clone();
            next_spec.set_pitch_apply_mode(name, call.call_position())?;
            let mode = next_spec.control.pitch.pitch_apply_mode;
            match population.status {
                PopulationStatus::Live => {
                    let mut proposed = population.spec.spawn_spec().control;
                    proposed.pitch.pitch_apply_mode = mode;
                    proposed.validate_pitch_application().map_err(|message| {
                        Box::new(EvalAltResult::ErrorRuntime(
                            message.into(),
                            call.call_position(),
                        ))
                    })?;
                    population.spec = next_spec;
                    population.pending_patch.pitch_apply_mode = Some(mode);
                }
                _ => ctx.warn_population_inactive(
                    handle.id,
                    "pitch_apply_mode",
                    call.call_position(),
                ),
            }
            Ok(handle)
        },
    );
    register_population_numeric_methods(
        engine,
        ctx.clone(),
        &[(
            "glide",
            PopulationSpec::set_pitch_glide_tau_sec,
            patch_pitch_glide_tau,
        )],
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "unison",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, value: INT| {
            spec.spec
                .record_call("unison", call.call_position(), &diagnostics);
            spec.spec.set_unison(value);
            spec
        },
    );
    let ctx_for_unison = ctx.clone();
    engine.register_fn(
        "unison",
        move |call: NativeCallContext, handle: PopulationHandle, value: INT| {
            let mut ctx = ctx_for_unison.lock().expect("lock script context");
            let diagnostics = ctx.warnings.clone();
            match ctx.populations.get_mut(&handle.id) {
                Some(population) if population.status == PopulationStatus::Live => {
                    population
                        .spec
                        .record_call("unison", call.call_position(), &diagnostics);
                    population.spec.set_unison(value);
                    population.pending_patch.timbre_unison = Some(value);
                }
                _ => ctx.warn_population_inactive(handle.id, "unison", call.call_position()),
            }
            handle
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "respawn_capacity",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, value: INT| {
            spec.spec
                .record_call("respawn_capacity", call.call_position(), &diagnostics);
            spec.spec.set_respawn_capacity(value);
            spec
        },
    );

    let ctx_for_boundary = ctx.clone();
    numeric_overloads!(engine, "rearticulate_above", |call, handle: PopulationHandle; value| {
        let mut ctx = ctx_for_boundary.lock().expect("lock script context");
        let position = call.call_position();
        ctx.record_harmonic_site(handle.id, "rearticulate_above", position);
        if let Some(population) = ctx.populations.get_mut(&handle.id) {
            if population.status != PopulationStatus::Live {
                ctx.warn_population_inactive(handle.id, "rearticulate_above", position);
            } else if population.spec.spawn_spec().control.pitch.pitch_apply_mode != PitchApplyMode::Rearticulate {
                ctx.warn("population_patch", "rearticulate_above() ignored outside pitch_apply_mode(\"rearticulate\")".into(), position);
            } else {
                population.spec.set_rearticulate_above(value);
                patch_rearticulate_above(&mut population.pending_patch, value);
            }
        } else {
            ctx.warn("population_patch", format!("rearticulate_above ignored for {} population {}", ctx.population_absence(handle.id), handle.id), position);
        }
        Ok::<_, Box<EvalAltResult>>(handle)
    });

    // Initial-only configuration is deliberately registered only on
    // PopulationSpec. A placed Population exposes live patches and release.

    register_population_numeric_methods(
        engine,
        ctx.clone(),
        &[
            (
                "brightness",
                PopulationSpec::set_brightness,
                patch_timbre_brightness,
            ),
            (
                "tail_roughness_aversion",
                PopulationSpec::set_tail_roughness_aversion,
                patch_tail_roughness_aversion,
            ),
            ("spread", PopulationSpec::set_spread, patch_timbre_spread),
        ],
    );
}

type PopulationPatchNumericSetter = fn(&mut ControlUpdate, f32);
type PopulationNumericMethod = (
    &'static str,
    PopulationSpecNumericSetter,
    PopulationPatchNumericSetter,
);

fn apply_population_numeric_patch(
    ctx_arc: &Arc<Mutex<ScriptContext>>,
    handle: PopulationHandle,
    label: &'static str,
    value: f32,
    spec_setter: PopulationSpecNumericSetter,
    patch_setter: PopulationPatchNumericSetter,
    position: Position,
) -> Result<PopulationHandle, Box<EvalAltResult>> {
    let mut ctx = ctx_arc.lock().expect("lock script context");
    let diagnostics = ctx.warnings.clone();
    ctx.record_harmonic_site(handle.id, label, position);
    let Some(population) = ctx.populations.get_mut(&handle.id) else {
        ctx.warn(
            "population_patch",
            format!(
                "{label} ignored for {} population {}",
                ctx.population_absence(handle.id),
                handle.id
            ),
            position,
        );
        return Ok(handle);
    };
    match population.status {
        PopulationStatus::Live => {
            population.spec.record_call(label, position, &diagnostics);
            spec_setter(&mut population.spec, value);
            patch_setter(&mut population.pending_patch, value);
        }
        PopulationStatus::ReleasePending | PopulationStatus::Released => {
            ctx.warn(
                "population_patch",
                format!("{label} ignored for inactive population {}", handle.id),
                position,
            );
        }
    }
    Ok(handle)
}

fn register_population_numeric_overloads(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    name: &'static str,
    spec_setter: PopulationSpecNumericSetter,
    patch_setter: PopulationPatchNumericSetter,
) {
    numeric_overloads!(engine, name, |call, handle: PopulationHandle; value| {
        apply_population_numeric_patch(&ctx, handle, name, value, spec_setter, patch_setter, call.call_position())
    });
}

fn register_population_numeric_methods(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    methods: &[PopulationNumericMethod],
) {
    for &(name, spec_setter, patch_setter) in methods {
        register_population_numeric_overloads(engine, ctx.clone(), name, spec_setter, patch_setter);
    }
}

fn apply_population_crowding(
    ctx_arc: &Arc<Mutex<ScriptContext>>,
    handle: PopulationHandle,
    label: &'static str,
    strength: f32,
    sigma_cents: Option<f32>,
    position: Position,
) -> Result<PopulationHandle, Box<EvalAltResult>> {
    let mut ctx = ctx_arc.lock().expect("lock script context");
    let Some(population) = ctx.populations.get_mut(&handle.id) else {
        ctx.warn(
            "population_patch",
            format!(
                "{label} ignored for {} population {}",
                ctx.population_absence(handle.id),
                handle.id
            ),
            position,
        );
        return Ok(handle);
    };
    match population.status {
        PopulationStatus::Live => match sigma_cents {
            Some(sigma) => {
                population.spec.set_crowding(strength, sigma);
                population.pending_patch.crowding_strength = Some(strength);
                population.pending_patch.crowding_sigma_cents = Some(sigma);
            }
            None => {
                population.spec.set_crowding_strength_only(strength);
                population.pending_patch.crowding_strength = Some(strength);
            }
        },
        _ => ctx.warn_population_inactive(handle.id, label, position),
    }
    Ok(handle)
}

fn register_population_crowding_overloads(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    name: &'static str,
) {
    let ctx_pair = ctx.clone();
    numeric_overloads!(engine, name, |call, handle: PopulationHandle; strength, sigma_cents| {
        apply_population_crowding(&ctx_pair, handle, name, strength, Some(sigma_cents), call.call_position())
    });
    numeric_overloads!(engine, name, |call, handle: PopulationHandle; strength| {
        apply_population_crowding(&ctx, handle, name, strength, None, call.call_position())
    });
}

fn patch_amp(update: &mut ControlUpdate, value: f32) {
    update.amp = Some(value);
}

fn patch_freq(update: &mut ControlUpdate, value: f32) {
    update.freq = Some(value);
}

fn patch_continuous_drive(update: &mut ControlUpdate, value: f32) {
    update.continuous_drive = Some(value);
}

fn patch_move_cost_coeff(update: &mut ControlUpdate, value: f32) {
    update.move_cost_coeff = Some(value);
}

fn patch_proposal_interval(update: &mut ControlUpdate, value: f32) {
    update.proposal_interval_sec = Some(value);
}

fn patch_temperature(update: &mut ControlUpdate, value: f32) {
    update.temperature = Some(value);
}

fn patch_tail_roughness_aversion(update: &mut ControlUpdate, value: f32) {
    update.tail_roughness_aversion = Some(crate::core::float::sanitize01(value));
}

fn patch_rearticulate_above(update: &mut ControlUpdate, value: f32) {
    update.rearticulate_above_cents = Some(value);
}

fn patch_pitch_glide_tau(update: &mut ControlUpdate, value: f32) {
    update.pitch_glide_tau_sec = Some(value);
}

fn patch_timbre_brightness(update: &mut ControlUpdate, value: f32) {
    update.timbre_brightness = Some(value);
}

fn patch_timbre_spread(update: &mut ControlUpdate, value: f32) {
    update.timbre_spread = Some(value);
}
