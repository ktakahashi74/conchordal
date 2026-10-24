// Expand numeric arguments consistently for every owner, including mixed pairs.
macro_rules! numeric_overloads {
    ($engine:expr, $name:expr, |$call:ident $(, $fixed:ident: $fixed_ty:ty)*; $($value:ident),+| $body:block $(,)?) => {{
        let operation = move |$call: NativeCallContext $(, $fixed: $fixed_ty)*, $($value: f32),+| $body;
        numeric_overloads!(@expand $engine, $name, operation, [$($fixed: $fixed_ty,)*], [], [$($value),+]);
    }};
    (@expand $engine:expr, $name:expr, $operation:ident, [$($fixed:ident: $fixed_ty:ty,)*], [$($done:ident: $ty:ty,)*], [$next:ident $(, $rest:ident)*]) => {
        numeric_overloads!(@expand $engine, $name, $operation, [$($fixed: $fixed_ty,)*], [$($done: $ty,)* $next: FLOAT,], [$($rest),*]);
        numeric_overloads!(@expand $engine, $name, $operation, [$($fixed: $fixed_ty,)*], [$($done: $ty,)* $next: INT,], [$($rest),*]);
    };
    (@expand $engine:expr, $name:expr, $operation:ident, [$($fixed:ident: $fixed_ty:ty,)*], [$($value:ident: $ty:ty,)*], []) => {{
        let operation = $operation.clone();
        $engine.register_fn($name, move |call: NativeCallContext $(, $fixed: $fixed_ty)*, $($value: $ty),*| operation(call $(, $fixed)*, $($value as f32),*));
    }};
}

use super::*;

impl ScriptHost {
    pub fn create_engine(ctx: Arc<Mutex<ScriptContext>>) -> Engine {
        let mut engine = Engine::new();
        engine.on_print(|msg| println!("[rhai] {msg}"));

        engine.register_type_with_name::<PopulationSpecHandle>("PopulationSpec");
        engine.register_type_with_name::<PopulationHandle>("Population");
        engine.register_type_with_name::<FloraSpecHandle>("FloraSpec");
        engine.register_type_with_name::<FloraHandle>("Flora");
        engine.register_type_with_name::<Placement>("Placement");
        engine.register_type_with_name::<Bus>("Bus");
        engine.register_type_with_name::<BusSet>("BusSet");
        engine.register_type_with_name::<ModePattern>("ModePattern");

        let mut builtins = rhai::Module::new();
        builtins.set_var("habitat_bus", Bus::habitat());
        builtins.set_var("presentation_bus", Bus::presentation());
        engine.register_global_module(builtins.into());

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

        engine.register_fn("flora", || FloraSpecHandle {
            spec: FloraSpec::default(),
        });
        for name in ["amp", "fade"] {
            engine.register_fn(
                name,
                move |call: NativeCallContext, spec: FloraSpecHandle, value: FLOAT| {
                    spec.with_numeric(name, value, call.call_position())
                },
            );
            engine.register_fn(
                name,
                move |call: NativeCallContext, spec: FloraSpecHandle, value: INT| {
                    spec.with_numeric(name, value as FLOAT, call.call_position())
                },
            );
        }
        {
            let name = "amp";
            let ctx_float = ctx.clone();
            engine.register_fn(
                name,
                move |call: NativeCallContext, handle: FloraHandle, value: FLOAT| {
                    let mut ctx = ctx_float.lock().expect("lock script context");
                    update_flora_numeric(&mut ctx, handle, name, value, call.call_position())
                },
            );
            let ctx_int = ctx.clone();
            engine.register_fn(
                name,
                move |call: NativeCallContext, handle: FloraHandle, value: INT| {
                    let mut ctx = ctx_int.lock().expect("lock script context");
                    update_flora_numeric(
                        &mut ctx,
                        handle,
                        name,
                        value as FLOAT,
                        call.call_position(),
                    )
                },
            );
        }
        engine.register_fn(
            "tone",
            |call: NativeCallContext,
             mut spec: FloraSpecHandle,
             name: &str|
             -> Result<FloraSpecHandle, Box<EvalAltResult>> {
                spec.spec.tone = parse_enum(
                    "Flora tone",
                    name,
                    call.call_position(),
                    &[("sine", FloraTone::Sine), ("resonant", FloraTone::Resonant)],
                )?;
                Ok(spec)
            },
        );
        engine.register_fn("residual", |mut spec: FloraSpecHandle, enabled: bool| {
            spec.spec.residual = enabled;
            spec
        });
        engine.register_fn("send", |mut spec: FloraSpecHandle, bus: Bus| {
            spec.spec.routing = bus.routing;
            spec
        });
        engine.register_fn("send", |mut spec: FloraSpecHandle, buses: BusSet| {
            spec.spec.routing = buses.routing();
            spec
        });
        let ctx_place_flora = ctx.clone();
        engine.register_fn(
            "place",
            move |call: NativeCallContext, spec: FloraSpecHandle, placement: Placement| {
                ctx_place_flora
                    .lock()
                    .expect("lock script context")
                    .place_flora(spec, placement, call.call_position())
            },
        );
        let ctx_release_flora = ctx.clone();
        engine.register_fn(
            "release",
            move |call: NativeCallContext, handle: FloraHandle| {
                ctx_release_flora
                    .lock()
                    .expect("lock script context")
                    .release_flora(handle.id, call.call_position());
            },
        );

        engine.register_fn("|", |left: Bus, right: Bus| left.set().combine(right.set()));
        engine.register_fn("|", |left: BusSet, right: Bus| left.combine(right.set()));
        engine.register_fn("|", |left: Bus, right: BusSet| left.set().combine(right));
        engine.register_fn("|", |left: BusSet, right: BusSet| left.combine(right));
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "send",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  bus: Bus| {
                population_spec
                    .spec
                    .record_call("send", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_routing(bus.set().routing());
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "send",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  bus: BusSet| {
                population_spec
                    .spec
                    .record_call("send", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_routing(bus.routing());
                population_spec
            },
        );

        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("amp", PopulationSpec::set_amp),
                ("freq", PopulationSpec::set_freq),
                ("landscape_weight", PopulationSpec::set_landscape_weight),
                (
                    "neighbor_step_cents",
                    PopulationSpec::set_neighbor_step_cents,
                ),
                ("tessitura_gravity", PopulationSpec::set_tessitura_gravity),
                ("sustain_drive", PopulationSpec::set_continuous_drive),
            ],
        );
        register_population_spec_pair_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[("avoid_neighbors", PopulationSpec::set_crowding)],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "avoid_neighbors",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  strength: FLOAT| {
                population_spec.spec.record_call(
                    "avoid_neighbors",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_crowding_strength_only(strength as f32);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "avoid_neighbors",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  strength: INT| {
                population_spec.spec.record_call(
                    "avoid_neighbors",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_crowding_strength_only(strength as f32);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "crowding_target",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  same_population_visible: bool,
                  other_population_visible: bool| {
                population_spec.spec.record_call(
                    "crowding_target",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_crowding_target(same_population_visible, other_population_visible);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "leave_self_out",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  enabled: bool| {
                population_spec.spec.record_call(
                    "leave_self_out",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_leave_self_out(enabled);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("move_cost", PopulationSpec::set_move_cost_coeff),
                ("move_cost_exp", PopulationSpec::set_move_cost_exp),
                (
                    "proposal_interval",
                    PopulationSpec::set_proposal_interval_sec,
                ),
            ],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "global_peaks",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  count: INT| {
                population_spec.spec.record_call(
                    "global_peaks",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_global_peaks(count, 0.0);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "global_peaks",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  count: INT,
                  min_sep_cents: FLOAT| {
                population_spec.spec.record_call(
                    "global_peaks",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_global_peaks(count, min_sep_cents as f32);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "global_peaks",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  count: INT,
                  min_sep_cents: INT| {
                population_spec.spec.record_call(
                    "global_peaks",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_global_peaks(count, min_sep_cents as f32);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "ratio_candidates",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  count: INT| {
                population_spec.spec.record_call(
                    "ratio_candidates",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_ratio_candidates(count);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("window_cents", PopulationSpec::set_window_cents),
                ("temperature", PopulationSpec::set_temperature),
                ("sigma_cents", PopulationSpec::set_sigma_cents),
            ],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "move_cost_time_scale",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  name: &str|
                  -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                population_spec.spec.record_call(
                    "move_cost_time_scale",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_move_cost_time_scale(name, call_ctx.call_position())?;
                Ok(population_spec)
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "pitch_apply_mode",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  name: &str|
                  -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                population_spec.spec.record_call(
                    "pitch_apply_mode",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_pitch_apply_mode(name, call_ctx.call_position())?;
                Ok(population_spec)
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("tie", PopulationSpec::set_tie),
                ("glide", PopulationSpec::set_pitch_glide_tau_sec),
                ("rearticulate_above", PopulationSpec::set_rearticulate_above),
                (
                    "articulation_persistence",
                    PopulationSpec::set_articulation_persistence,
                ),
            ],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "seek_consonance",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec.spec.record_call(
                    "seek_consonance",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_consonance_movement();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "anchor",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec
                    .spec
                    .record_call("anchor", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_anchor();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "pitch_core",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  name: &str|
                  -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                population_spec.spec.record_call(
                    "pitch_core",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_pitch_core(name, call_ctx.call_position())?;
                Ok(population_spec)
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "brain",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  name: &str|
                  -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                population_spec
                    .spec
                    .record_call("brain", call_ctx.call_position(), &diagnostics);
                population_spec
                    .spec
                    .set_brain(name, call_ctx.call_position())?;
                Ok(population_spec)
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "sustain",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec
                    .spec
                    .record_call("sustain", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_phonation(PhonationKind::Sustain);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "repeat",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec
                    .spec
                    .record_call("repeat", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_phonation(PhonationKind::Repeat);
                population_spec
            },
        );
        // Timing presets separate shared-beat synchronization from private participation.
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "metric",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec
                    .spec
                    .record_call("metric", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_metric();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "entrained",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec.spec.record_call(
                    "entrained",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_entrained();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "steady",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec
                    .spec
                    .record_call("steady", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_steady();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "flow",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec
                    .spec
                    .record_call("flow", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_flow();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "intrinsic_rate",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  hz: FLOAT| {
                population_spec.spec.record_call(
                    "intrinsic_rate",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.intrinsic_rate = Some(hz);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "intrinsic_rate",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  hz: INT| {
                population_spec.spec.record_call(
                    "intrinsic_rate",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.intrinsic_rate = Some(hz as FLOAT);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "rhythm_role",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  name: &str|
                  -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                population_spec.spec.record_call(
                    "rhythm_role",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_rhythm_role(name, call_ctx.call_position())?;
                Ok(population_spec)
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("entrainment", PopulationSpec::set_entrainment),
                ("microtiming", PopulationSpec::set_microtiming),
                ("measure_accent", PopulationSpec::set_measure_accent),
            ],
        );
        // Tier 2: explicit when/duration
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "once",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec
                    .spec
                    .record_call("once", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_when_once();
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[("pulse", PopulationSpec::set_when_pulse)],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "while_alive",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec.spec.record_call(
                    "while_alive",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_duration_while_alive();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "phonate_when_viable",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec.spec.record_call(
                    "phonate_when_viable",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_phonate_when_viable();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "cycles",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  n: INT| {
                population_spec
                    .spec
                    .record_call("cycles", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_duration_cycles(n.max(1) as u32);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "adaptive_duration",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec.spec.record_call(
                    "adaptive_duration",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_adaptive_duration();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "rest",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec
                    .spec
                    .record_call("rest", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_rest(1.0);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[("rest", PopulationSpec::set_rest)],
        );
        // Tier 3: expert tuning
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("pulse_lock", PopulationSpec::set_pulse_lock),
                ("social", PopulationSpec::set_social),
                ("shorten_on_drop", PopulationSpec::set_shorten_on_drop),
            ],
        );
        register_population_spec_pair_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("duration_range", PopulationSpec::set_duration_range),
                ("duration_curve", PopulationSpec::set_duration_curve),
            ],
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("brightness", PopulationSpec::set_brightness),
                ("spread", PopulationSpec::set_spread),
            ],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "modes",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  pattern: ModePattern| {
                population_spec
                    .spec
                    .record_call("modes", call_ctx.call_position(), &diagnostics);
                population_spec.spec.set_modes(pattern);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("endurance", PopulationSpec::set_endurance),
                ("recovery", PopulationSpec::set_recovery),
                (
                    "attack_cost_fraction",
                    PopulationSpec::set_attack_cost_fraction,
                ),
                (
                    "attack_recharge_fraction",
                    PopulationSpec::set_attack_recharge_fraction,
                ),
            ],
        );
        register_population_spec_pair_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[(
                "consonance_viability",
                PopulationSpec::set_consonance_viability,
            )],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "viability_scope",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  name: &str|
                  -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                population_spec.spec.record_call(
                    "viability_scope",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_viability_scope(name, call_ctx.call_position())?;
                Ok(population_spec)
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[("dissonance_penalty", PopulationSpec::set_dissonance_penalty)],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        numeric_overloads!(engine, "adsr", |call_ctx, population_spec: PopulationSpecHandle; a, d, level, r| {
            let mut population_spec = population_spec;
            population_spec.spec.record_call("adsr", call_ctx.call_position(), &diagnostics);
            population_spec.spec.set_adsr(a, d, level, r);
            population_spec
        });
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        numeric_overloads!(engine, "rhythm_coupling_vitality", |call_ctx, population_spec: PopulationSpecHandle; lambda_v, v_floor| {
            let mut population_spec = population_spec;
            population_spec.spec.record_call("rhythm_coupling_vitality", call_ctx.call_position(), &diagnostics);
            population_spec.spec.set_rhythm_coupling_vitality(lambda_v, v_floor);
            population_spec
        });
        macro_rules! reward_overload {
            ($ty:ty) => {
                let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
                engine.register_fn(
                    "rhythm_reward",
                    move |call_ctx: NativeCallContext,
                          mut population_spec: PopulationSpecHandle,
                          rho_t: $ty,
                          metric: &str|
                          -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                        population_spec.spec.record_call(
                            "rhythm_reward",
                            call_ctx.call_position(),
                            &diagnostics,
                        );
                        population_spec.spec.set_rhythm_reward(
                            rho_t as f32,
                            metric,
                            call_ctx.call_position(),
                        )?;
                        Ok(population_spec)
                    },
                );
            };
        }
        reward_overload!(FLOAT);
        reward_overload!(INT);
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[("rhythm_freq", PopulationSpec::set_rhythm_freq)],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "respawn_random",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec.spec.record_call(
                    "respawn_random",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_respawn_random();
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "respawn_hereditary",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  sigma_oct: FLOAT| {
                population_spec.spec.record_call(
                    "respawn_hereditary",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_respawn_hereditary(sigma_oct as f32);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "respawn_hereditary",
            move |call_ctx: NativeCallContext,
                  mut population_spec: PopulationSpecHandle,
                  sigma_oct: INT| {
                population_spec.spec.record_call(
                    "respawn_hereditary",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec
                    .spec
                    .set_respawn_hereditary(sigma_oct as f32);
                population_spec
            },
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "respawn_consonance",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle| {
                population_spec.spec.record_call(
                    "respawn_consonance",
                    call_ctx.call_position(),
                    &diagnostics,
                );
                population_spec.spec.set_respawn_consonance();
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                (
                    "respawn_min_consonance",
                    PopulationSpec::set_respawn_min_c_level,
                ),
                (
                    "respawn_background_death_rate",
                    PopulationSpec::set_respawn_background_death_rate,
                ),
            ],
        );
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "respawn_settle",
            move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle, placement: Placement| {
                population_spec.spec.record_call("respawn_settle", call_ctx.call_position(), &diagnostics);
                if let Some(strategy) = placement.strategy() {
                    population_spec.spec.set_respawn_settle_strategy(strategy);
                } else {
                    diagnostics.lock().expect("lock script warnings").push(ScriptWarning {message: "respawn_settle() requires consonance(), dissonance(), edge(), gap(), random(), or line()".to_string(), position: call_ctx.call_position()});
                }
                population_spec
            },
        );

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
                ctx.wait(sec as f32);
                Ok(())
            },
        );
        let ctx_for_wait_int = ctx.clone();
        engine.register_fn(
            "wait",
            move |_call_ctx: NativeCallContext, sec: INT| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_wait_int.lock().expect("lock script context");
                ctx.wait(sec as f32);
                Ok(())
            },
        );

        let ctx_for_flush = ctx.clone();
        engine.register_fn(
            "flush",
            move |_call_ctx: NativeCallContext| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_flush.lock().expect("lock script context");
                ctx.flush();
                Ok(())
            },
        );

        let ctx_for_temporal_mode = ctx.clone();
        engine.register_fn(
            "temporal_mode",
            move |call: NativeCallContext, name: &str| -> Result<(), Box<EvalAltResult>> {
                let mut ctx = ctx_for_temporal_mode.lock().expect("lock script context");
                if ctx.temporal_mode_declared
                    || ctx.cursor != 0.0
                    || !ctx.scopes.is_empty()
                    || !ctx.scenario.events.is_empty()
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
                ctx.scenario.temporal_mode = mode;
                ctx.temporal_mode_declared = true;
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

        let ctx_for_section = ctx.clone();
        engine.register_fn(
            "section",
            move |call_ctx: NativeCallContext, name: &str, callback: FnPtr| {
                {
                    let mut ctx = ctx_for_section.lock().expect("lock script context");
                    ctx.push_scene_marker(name);
                    ctx.push_scope();
                }
                let result = callback.call_within_context::<Dynamic>(&call_ctx, ());
                let mut ctx = ctx_for_section.lock().expect("lock script context");
                ctx.pop_scope();
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
                ctx.pop_scope();
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
                ctx.pop_scope();
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
                ctx.pop_scope();
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
                ctx.pop_scope();
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
                ctx.pop_scope();
                result.map(|_| ())
            },
        );

        let ctx_for_parallel = ctx.clone();
        engine.register_fn(
            "parallel",
            move |call_ctx: NativeCallContext,
                  callbacks: Array|
                  -> Result<(), Box<EvalAltResult>> {
                let start_time = {
                    let ctx = ctx_for_parallel.lock().expect("lock script context");
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
                    ctx.pop_scope();
                    max_end = max_end.max(end_time);
                    let _ = result?;
                }
                let mut ctx = ctx_for_parallel.lock().expect("lock script context");
                ctx.cursor = max_end;
                Ok(())
            },
        );

        engine.register_fn("harmonic_modes", ModePattern::harmonic_modes);
        engine.register_fn("odd_modes", ModePattern::odd_modes);
        numeric_overloads!(engine, "power_modes", |_call; beta| { ModePattern::power_modes(beta) });
        numeric_overloads!(engine, "stiff_string_modes", |_call; stiffness| { ModePattern::stiff_string_modes(stiffness) });
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "custom_modes",
            move |call: NativeCallContext, ratios: Array| {
                ModePattern::custom_modes(rhai_array_to_f32(
                    ratios,
                    "custom_modes",
                    &diagnostics,
                    call.call_position(),
                ))
            },
        );
        engine.register_fn(
            "modal_table",
            |call: NativeCallContext, name: &str| -> Result<ModePattern, Box<EvalAltResult>> {
                let name = parse_enum(
                    "modal_table",
                    name,
                    call.call_position(),
                    &[
                        ("uniform_aluminum_bar", "uniform_aluminum_bar"),
                        ("uniform_wooden_bar", "uniform_wooden_bar"),
                        ("xylophone", "xylophone"),
                        ("vibraphone_1", "vibraphone_1"),
                        ("vibraphone_2", "vibraphone_2"),
                        ("wine_glass", "wine_glass"),
                    ],
                )?;
                Ok(ModePattern::modal_table(name).expect("validated modal table"))
            },
        );
        engine.register_fn(
            "landscape_density_modes",
            ModePattern::landscape_density_modes,
        );
        engine.register_fn("landscape_peaks_modes", ModePattern::landscape_peaks_modes);
        engine.register_fn("count", |pattern: ModePattern, n: INT| {
            pattern.with_count((n as usize).max(1))
        });
        let warnings = ctx.lock().unwrap().warnings.clone();
        numeric_overloads!(engine, "range", |call, pattern: ModePattern; min_mul, max_mul| { if pattern.supports_range() {pattern.with_range(min_mul, max_mul)} else {warnings.lock().unwrap().push(ScriptWarning {message: "range() is only supported for landscape_*_modes(); ignored".into(), position: call.call_position()}); pattern} });
        let warnings = ctx.lock().unwrap().warnings.clone();
        numeric_overloads!(engine, "spacing", |call, pattern: ModePattern; min_dist| { if pattern.supports_min_dist_erb() {pattern.with_min_dist_erb(min_dist)} else {warnings.lock().unwrap().push(ScriptWarning {message: "spacing() is only supported for landscape_*_modes(); ignored".into(), position: call.call_position()}); pattern} });
        let warnings = ctx.lock().unwrap().warnings.clone();
        numeric_overloads!(engine, "gamma", |call, pattern: ModePattern; gamma| { if pattern.supports_gamma() {pattern.with_gamma(gamma)} else {warnings.lock().unwrap().push(ScriptWarning {message: "gamma() is only supported for landscape_density_modes(); ignored".into(), position: call.call_position()}); pattern} });
        numeric_overloads!(engine, "jitter", |_call, pattern: ModePattern; cents| { pattern.with_jitter_cents(cents) });
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "seed",
            move |call_ctx: NativeCallContext, pattern: ModePattern, seed: INT| {
                if seed < 0 {
                    diagnostics
                        .lock()
                        .expect("lock script warnings")
                        .push(ScriptWarning {
                            message: "seed() expects >= 0".to_string(),
                            position: call_ctx.call_position(),
                        });
                    pattern
                } else {
                    pattern.with_seed(seed as u64)
                }
            },
        );

        numeric_overloads!(engine, "at", |_call; freq| {Placement::at(freq)});
        numeric_overloads!(engine, "consonance", |_call; root| {Placement::consonance_root(root)});
        for (name, constructor) in [
            (
                "consonance",
                Placement::consonance_range as fn(f32, f32) -> Placement,
            ),
            ("dissonance", Placement::dissonance),
            ("edge", Placement::edge),
            ("gap", Placement::gap),
            ("random", Placement::random),
            ("line", Placement::line),
        ] {
            numeric_overloads!(engine, name, |_call; lo, hi| {constructor(lo, hi)});
        }
        for (name, sampling) in [
            ("peak", FieldSampling::Peak),
            ("density", FieldSampling::Density),
        ] {
            let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
            engine.register_fn(
                name,
                move |call: NativeCallContext, mut placement: Placement| {
                    placement.diagnostics = diagnostics.clone();
                    placement.position = call.call_position();
                    placement.with_sampling(sampling)
                },
            );
        }
        for (name, setter) in [
            (
                "tension",
                Placement::with_tension as fn(Placement, f32) -> Placement,
            ),
            ("spacing", Placement::with_spacing),
        ] {
            let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
            numeric_overloads!(engine, name, |call, placement: Placement; value| {
                let mut placement = placement;
                placement.diagnostics = diagnostics.clone();
                placement.position = call.call_position();
                setter(placement, value)
            });
        }
        engine.register_fn("count", |placement: Placement, count: INT| {
            placement.with_count(count)
        });
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        numeric_overloads!(engine, "range", |call, placement: Placement; min_mul, max_mul| {
            let mut placement = placement;
            placement.diagnostics = diagnostics.clone();
            placement.position = call.call_position();
            placement.with_range(min_mul, max_mul)
        });

        register_population_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                ("amp", PopulationSpec::set_amp, patch_amp),
                ("freq", PopulationSpec::set_freq, patch_freq),
                (
                    "landscape_weight",
                    PopulationSpec::set_landscape_weight,
                    patch_landscape_weight,
                ),
                (
                    "neighbor_step_cents",
                    PopulationSpec::set_neighbor_step_cents,
                    patch_neighbor_step_cents,
                ),
                (
                    "tessitura_gravity",
                    PopulationSpec::set_tessitura_gravity,
                    patch_tessitura_gravity,
                ),
                (
                    "sustain_drive",
                    PopulationSpec::set_continuous_drive,
                    patch_continuous_drive,
                ),
            ],
        );
        register_population_crowding_overloads(&mut engine, ctx.clone(), "avoid_neighbors");
        let ctx_for_population_crowding_target = ctx.clone();
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        engine.register_fn(
            "crowding_target",
            move |call_ctx: NativeCallContext,
                  handle: PopulationHandle,
                  same_population_visible: bool,
                  other_population_visible: bool|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_crowding_target
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    diagnostics
                        .lock()
                        .expect("lock script warnings")
                        .push(ScriptWarning {
                            message: format!(
                                "crowding_target ignored for unknown population {}",
                                handle.id
                            ),
                            position: call_ctx.call_position(),
                        });
                    return Ok(handle);
                };
                match population.status {
                    PopulationStatus::Live => {
                        population.crowding_target_same = same_population_visible;
                        population.crowding_target_other = other_population_visible;
                        population
                            .spec
                            .set_crowding_target(same_population_visible, other_population_visible);
                        population.pending_crowding_target =
                            Some((same_population_visible, other_population_visible));
                    }
                    _ => ctx.warn_population_inactive(
                        handle.id,
                        "crowding_target",
                        call_ctx.call_position(),
                    ),
                }
                Ok(handle)
            },
        );
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
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    diagnostics
                        .lock()
                        .expect("lock script warnings")
                        .push(ScriptWarning {
                            message: format!(
                                "leave_self_out ignored for unknown population {}",
                                handle.id
                            ),
                            position: call_ctx.call_position(),
                        });
                    return Ok(handle);
                };
                match population.status {
                    PopulationStatus::Live => {
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
            &mut engine,
            ctx.clone(),
            &[
                (
                    "move_cost",
                    PopulationSpec::set_move_cost_coeff,
                    patch_move_cost_coeff,
                ),
                (
                    "move_cost_exp",
                    PopulationSpec::set_move_cost_exp,
                    patch_move_cost_exp,
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
                            message: format!(
                                "global_peaks ignored for unknown population {}",
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
                            message: format!(
                                "global_peaks ignored for unknown population {}",
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
                            message: format!(
                                "global_peaks ignored for unknown population {}",
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
                            message: format!(
                                "ratio_candidates ignored for unknown population {}",
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
            &mut engine,
            ctx.clone(),
            &[
                (
                    "window_cents",
                    PopulationSpec::set_window_cents,
                    patch_window_cents,
                ),
                (
                    "temperature",
                    PopulationSpec::set_temperature,
                    patch_temperature,
                ),
                (
                    "sigma_cents",
                    PopulationSpec::set_sigma_cents,
                    patch_sigma_cents,
                ),
            ],
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
                            message: format!(
                                "move_cost_time_scale ignored for unknown population {}",
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
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    diagnostics
                        .lock()
                        .expect("lock script warnings")
                        .push(ScriptWarning {
                            message: format!(
                                "pitch_apply_mode ignored for unknown population {}",
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
            &mut engine,
            ctx.clone(),
            &[(
                "glide",
                PopulationSpec::set_pitch_glide_tau_sec,
                patch_pitch_glide_tau,
            )],
        );
        for (name, spec_setter, patch_setter) in [
            (
                "unison",
                PopulationSpec::set_unison as fn(&mut PopulationSpec, INT),
                patch_timbre_unison as fn(&mut ControlUpdate, INT),
            ),
            ("top_k", PopulationSpec::set_top_k, patch_top_k),
            (
                "random_candidates",
                PopulationSpec::set_random_candidates,
                patch_random_candidates,
            ),
        ] {
            let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
            engine.register_fn(
                name,
                move |call: NativeCallContext, mut spec: PopulationSpecHandle, value: INT| {
                    spec.spec
                        .record_call(name, call.call_position(), &diagnostics);
                    spec_setter(&mut spec.spec, value);
                    spec
                },
            );
            let ctx_for_count = ctx.clone();
            engine.register_fn(
                name,
                move |call: NativeCallContext, handle: PopulationHandle, value: INT| {
                    let mut ctx = ctx_for_count.lock().expect("lock script context");
                    let diagnostics = ctx.warnings.clone();
                    match ctx.populations.get_mut(&handle.id) {
                        Some(population) if population.status == PopulationStatus::Live => {
                            population
                                .spec
                                .record_call(name, call.call_position(), &diagnostics);
                            spec_setter(&mut population.spec, value);
                            patch_setter(&mut population.pending_patch, value);
                        }
                        _ => ctx.warn_population_inactive(handle.id, name, call.call_position()),
                    }
                    handle
                },
            );
        }
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
            if let Some(population) = ctx.populations.get_mut(&handle.id) {
                if population.status != PopulationStatus::Live {
                    ctx.warn_population_inactive(handle.id, "rearticulate_above", position);
                } else if population.spec.spawn_spec().control.pitch.pitch_apply_mode != PitchApplyMode::Rearticulate {
                    ctx.warn("rearticulate_above() ignored outside pitch_apply_mode(\"rearticulate\")".into(), position);
                } else {
                    population.spec.set_rearticulate_above(value);
                    patch_rearticulate_above(&mut population.pending_patch, value);
                }
            } else {
                ctx.warn(format!("rearticulate_above ignored for unknown population {}", handle.id), position);
            }
            Ok::<_, Box<EvalAltResult>>(handle)
        });

        // Initial-only configuration is deliberately registered only on
        // PopulationSpec. A placed Population exposes live patches and release.

        register_population_numeric_methods(
            &mut engine,
            ctx.clone(),
            &[
                (
                    "brightness",
                    PopulationSpec::set_brightness,
                    patch_timbre_brightness,
                ),
                ("spread", PopulationSpec::set_spread, patch_timbre_spread),
            ],
        );

        // Director-level shaping of the emergent production meter (scene-global
        // soft priors, symmetric to the consonance-field ops). These never
        // schedule a beat: `meter_stability` sets how readily a pulse forms,
        // `temporal_basin` sets the tempo region the pulse gravitates toward.
        let ctx_for_meter_stability_f = ctx.clone();
        engine.register_fn(
            "meter_stability",
            move |_call_ctx: NativeCallContext, value: FLOAT| {
                let mut ctx = ctx_for_meter_stability_f
                    .lock()
                    .expect("lock script context");
                ctx.scenario.meter_shaping.stability = (value as f32).clamp(0.0, 1.0);
            },
        );
        let ctx_for_meter_stability_i = ctx.clone();
        engine.register_fn(
            "meter_stability",
            move |_call_ctx: NativeCallContext, value: INT| {
                let mut ctx = ctx_for_meter_stability_i
                    .lock()
                    .expect("lock script context");
                ctx.scenario.meter_shaping.stability = (value as f32).clamp(0.0, 1.0);
            },
        );
        macro_rules! register_temporal_basin {
            ($min_ty:ty, $max_ty:ty) => {{
                let ctx_clone = ctx.clone();
                engine.register_fn(
                    "temporal_basin",
                    move |_call_ctx: NativeCallContext, min: $min_ty, max: $max_ty| {
                        let mut ctx = ctx_clone.lock().expect("lock script context");
                        ctx.scenario.meter_shaping.basin_hz = Some((min as f32, max as f32));
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
                ctx.push_event(cursor, vec![Action::UpdateLandscape { update }]);
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
                ctx_for_set_control_update_mode
                    .lock()
                    .expect("lock script context")
                    .scenario
                    .control_update_mode = mode;
                Ok(())
            },
        );

        let ctx_for_set_scaffold_off = ctx.clone();
        engine.register_fn("scaffold_off", move |_call_ctx: NativeCallContext| {
            let mut ctx = ctx_for_set_scaffold_off
                .lock()
                .expect("lock script context");
            ctx.scenario.scaffold = ScaffoldConfig::Off;
        });
        let ctx_for_set_scaffold_shared = ctx.clone();
        engine.register_fn(
            "scaffold_shared",
            move |_call_ctx: NativeCallContext, freq_hz: FLOAT| {
                let mut ctx = ctx_for_set_scaffold_shared
                    .lock()
                    .expect("lock script context");
                ctx.scenario.scaffold = ScaffoldConfig::Shared {
                    freq_hz: (freq_hz as f32).max(0.0),
                };
            },
        );
        let ctx_for_set_scaffold_shared_int = ctx.clone();
        engine.register_fn(
            "scaffold_shared",
            move |_call_ctx: NativeCallContext, freq_hz: INT| {
                let mut ctx = ctx_for_set_scaffold_shared_int
                    .lock()
                    .expect("lock script context");
                ctx.scenario.scaffold = ScaffoldConfig::Shared {
                    freq_hz: (freq_hz as f32).max(0.0),
                };
            },
        );
        let ctx_for_set_scaffold_scrambled = ctx.clone();
        engine.register_fn(
            "scaffold_scrambled",
            move |_call_ctx: NativeCallContext, freq_hz: FLOAT, seed: INT| {
                let mut ctx = ctx_for_set_scaffold_scrambled
                    .lock()
                    .expect("lock script context");
                ctx.scenario.scaffold = ScaffoldConfig::Scrambled {
                    freq_hz: (freq_hz as f32).max(0.0),
                    seed: seed.max(0) as u64,
                };
            },
        );
        let ctx_for_set_scaffold_scrambled_int = ctx.clone();
        engine.register_fn(
            "scaffold_scrambled",
            move |_call_ctx: NativeCallContext, freq_hz: INT, seed: INT| {
                let mut ctx = ctx_for_set_scaffold_scrambled_int
                    .lock()
                    .expect("lock script context");
                ctx.scenario.scaffold = ScaffoldConfig::Scrambled {
                    freq_hz: (freq_hz as f32).max(0.0),
                    seed: seed.max(0) as u64,
                };
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
                );
            },
        );

        engine
    }

    /// Loads and evaluates a scenario script. `seed`, when given, seeds the
    /// fresh scenario before evaluation; a script-level `seed()` call still
    /// wins, since it runs during evaluation and overwrites it.
    pub fn load_script(path: &str, seed: Option<u64>) -> Result<Scenario, ScriptError> {
        let (scenario, warnings) = Self::load_script_with_warnings(path, seed)?;
        for warning in warnings {
            eprintln!(
                "{path}:{:?}: warning: {}",
                warning.position, warning.message
            );
        }
        Ok(scenario)
    }

    /// Evaluate the script and return positioned diagnostics to CLI and test consumers.
    pub fn load_script_with_warnings(
        path: &str,
        seed: Option<u64>,
    ) -> Result<(Scenario, Vec<ScriptWarning>), ScriptError> {
        let src = fs::read_to_string(path)
            .map_err(|err| ScriptError::new(format!("{path}: {err}"), None))?;
        let ctx = Arc::new(Mutex::new(match seed {
            Some(seed) => ScriptContext::with_seed(seed),
            None => ScriptContext::default(),
        }));
        let engine = Self::create_engine(ctx.clone());
        engine
            .eval::<()>(&src)
            .map_err(|err| ScriptError::from_eval(err, Some(path)))?;
        let mut ctx = ctx.lock().expect("lock script context");
        ctx.finish();
        let warnings = ctx.warnings.lock().expect("lock script warnings").clone();
        Ok((ctx.scenario.clone(), warnings))
    }
}

type PopulationSpecNumericSetter = fn(&mut PopulationSpec, f32);
type PopulationSpecPairNumericSetter = fn(&mut PopulationSpec, f32, f32);
type PopulationPatchNumericSetter = fn(&mut ControlUpdate, f32);
type PopulationSpecNumericMethod = (&'static str, PopulationSpecNumericSetter);
type PopulationSpecPairNumericMethod = (&'static str, PopulationSpecPairNumericSetter);
type PopulationNumericMethod = (
    &'static str,
    PopulationSpecNumericSetter,
    PopulationPatchNumericSetter,
);

fn register_population_spec_numeric_overloads(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    name: &'static str,
    setter: PopulationSpecNumericSetter,
) {
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    numeric_overloads!(engine, name, |call, population_spec: PopulationSpecHandle; value| {
        let mut population_spec = population_spec;
        population_spec.spec.record_call(name, call.call_position(), &diagnostics);
        setter(&mut population_spec.spec, value);
        population_spec
    });
}

fn register_population_spec_pair_numeric_overloads(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    name: &'static str,
    setter: PopulationSpecPairNumericSetter,
) {
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    numeric_overloads!(engine, name, |call, population_spec: PopulationSpecHandle; first, second| {
        let mut population_spec = population_spec;
        population_spec.spec.record_call(name, call.call_position(), &diagnostics);
        setter(&mut population_spec.spec, first, second);
        population_spec
    });
}

fn register_population_spec_numeric_methods(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    methods: &[PopulationSpecNumericMethod],
) {
    for &(name, setter) in methods {
        register_population_spec_numeric_overloads(engine, ctx.clone(), name, setter);
    }
}

fn register_population_spec_pair_numeric_methods(
    engine: &mut Engine,
    ctx: Arc<Mutex<ScriptContext>>,
    methods: &[PopulationSpecPairNumericMethod],
) {
    for &(name, setter) in methods {
        register_population_spec_pair_numeric_overloads(engine, ctx.clone(), name, setter);
    }
}

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
    let Some(population) = ctx.populations.get_mut(&handle.id) else {
        ctx.warn(
            format!("{label} ignored for unknown population {}", handle.id),
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
            format!("{label} ignored for unknown population {}", handle.id),
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

fn patch_landscape_weight(update: &mut ControlUpdate, value: f32) {
    update.landscape_weight = Some(value);
}

fn patch_neighbor_step_cents(update: &mut ControlUpdate, value: f32) {
    update.neighbor_step_cents = Some(value);
}

fn patch_tessitura_gravity(update: &mut ControlUpdate, value: f32) {
    update.tessitura_gravity = Some(value);
}

fn patch_continuous_drive(update: &mut ControlUpdate, value: f32) {
    update.continuous_drive = Some(value);
}

fn patch_move_cost_coeff(update: &mut ControlUpdate, value: f32) {
    update.move_cost_coeff = Some(value);
}

fn patch_move_cost_exp(update: &mut ControlUpdate, value: f32) {
    update.move_cost_exp = Some(value.round() as i64);
}

fn patch_proposal_interval(update: &mut ControlUpdate, value: f32) {
    update.proposal_interval_sec = Some(value);
}

fn patch_window_cents(update: &mut ControlUpdate, value: f32) {
    update.window_cents = Some(value);
}

fn patch_top_k(update: &mut ControlUpdate, value: INT) {
    update.top_k = Some(value);
}

fn patch_temperature(update: &mut ControlUpdate, value: f32) {
    update.temperature = Some(value);
}

fn patch_sigma_cents(update: &mut ControlUpdate, value: f32) {
    update.sigma_cents = Some(value);
}

fn patch_random_candidates(update: &mut ControlUpdate, value: INT) {
    update.random_candidates = Some(value);
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

fn patch_timbre_unison(update: &mut ControlUpdate, value: INT) {
    update.timbre_unison = Some(value);
}
