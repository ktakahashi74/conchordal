use super::{
    Arc, Bus, BusSet, Engine, EvalAltResult, FLOAT, INT, ModePattern, Mutex, NativeCallContext,
    PhonationKind, Placement, PopulationSpec, PopulationSpecHandle, ScriptContext, ScriptWarning,
};

pub(super) fn register_spec_methods(engine: &mut Engine, ctx: &Arc<Mutex<ScriptContext>>) {
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "send",
        move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle, bus: Bus| {
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
        engine,
        ctx.clone(),
        &[
            ("amp", PopulationSpec::set_amp),
            ("freq", PopulationSpec::set_freq),
            ("sustain_drive", PopulationSpec::set_continuous_drive),
        ],
    );
    register_population_spec_pair_numeric_methods(
        engine,
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
        engine,
        ctx.clone(),
        &[
            ("move_cost", PopulationSpec::set_move_cost_coeff),
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
        engine,
        ctx.clone(),
        &[("temperature", PopulationSpec::set_temperature)],
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
        engine,
        ctx.clone(),
        &[
            (
                "tail_roughness_aversion",
                PopulationSpec::set_tail_roughness_aversion,
            ),
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
            population_spec
                .spec
                .record_call("pitch_core", call_ctx.call_position(), &diagnostics);
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
            population_spec
                .spec
                .record_call("entrained", call_ctx.call_position(), &diagnostics);
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
        move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle, hz: FLOAT| {
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
        move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle, hz: INT| {
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
            population_spec
                .spec
                .record_call("rhythm_role", call_ctx.call_position(), &diagnostics);
            population_spec
                .spec
                .set_rhythm_role(name, call_ctx.call_position())?;
            Ok(population_spec)
        },
    );
    register_population_spec_numeric_methods(
        engine,
        ctx.clone(),
        &[
            ("entrainment", PopulationSpec::set_entrainment),
            ("microtiming", PopulationSpec::set_microtiming),
            ("measure_accent", PopulationSpec::set_measure_accent),
            #[cfg(feature = "research")]
            ("syncopate", PopulationSpec::set_syncopate),
        ],
    );
    #[cfg(not(feature = "research"))]
    {
        let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
        let register = move |call: NativeCallContext, mut spec: PopulationSpecHandle, nonzero| {
            if nonzero {
                return Err(EvalAltResult::ErrorRuntime(
                    "syncopate requires the research feature".into(),
                    call.call_position(),
                )
                .into());
            }
            spec.spec
                .record_call("syncopate", call.call_position(), &diagnostics);
            Ok::<_, Box<EvalAltResult>>(spec)
        };
        let register_int = register.clone();
        engine.register_fn(
            "syncopate",
            move |call: NativeCallContext, spec: PopulationSpecHandle, value: INT| {
                register_int(call, spec, value != 0)
            },
        );
        engine.register_fn(
            "syncopate",
            move |call: NativeCallContext, spec: PopulationSpecHandle, value: FLOAT| {
                register(call, spec, value != 0.0)
            },
        );
    }
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
        engine,
        ctx.clone(),
        &[("pulse", PopulationSpec::set_when_pulse)],
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
        move |call_ctx: NativeCallContext, mut population_spec: PopulationSpecHandle, n: INT| {
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
        engine,
        ctx.clone(),
        &[
            ("rest", PopulationSpec::set_rest),
            (
                "rest_on_habituation",
                PopulationSpec::set_rest_on_habituation,
            ),
            ("sway", PopulationSpec::set_sway),
        ],
    );
    // Clamp finite Rhai floats before narrowing to the DSP representation.
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "dwell",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: FLOAT| {
            spec.spec
                .record_call("dwell", call.call_position(), &diagnostics);
            spec.spec.set_dwell(weight);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "dwell",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: INT| {
            spec.spec
                .record_call("dwell", call.call_position(), &diagnostics);
            spec.spec.set_dwell(weight as FLOAT);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "respond_to_onsets",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: FLOAT| {
            spec.spec
                .record_call("respond_to_onsets", call.call_position(), &diagnostics);
            spec.spec.set_respond_to_onsets(weight);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "respond_to_onsets",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: INT| {
            spec.spec
                .record_call("respond_to_onsets", call.call_position(), &diagnostics);
            spec.spec.set_respond_to_onsets(weight as FLOAT);
            spec
        },
    );
    // Clamp finite Rhai floats before narrowing to the DSP representation.
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "follow_own_sequence",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: FLOAT| {
            spec.spec
                .record_call("follow_own_sequence", call.call_position(), &diagnostics);
            spec.spec.set_follow_own_sequence(weight);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "follow_own_sequence",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: INT| {
            spec.spec
                .record_call("follow_own_sequence", call.call_position(), &diagnostics);
            spec.spec.set_follow_own_sequence(weight as FLOAT);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "phrase_boundary",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: FLOAT| {
            spec.spec
                .record_call("phrase_boundary", call.call_position(), &diagnostics);
            spec.spec.set_phrase_boundary(weight);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "follow_tonal_context",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: FLOAT| {
            spec.spec
                .record_call("follow_tonal_context", call.call_position(), &diagnostics);
            spec.spec.set_follow_tonal_context(weight);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "melodic_continuity",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: FLOAT| {
            spec.spec
                .record_call("melodic_continuity", call.call_position(), &diagnostics);
            spec.spec.set_melodic_continuity(weight);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "phrase_boundary",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: INT| {
            spec.spec
                .record_call("phrase_boundary", call.call_position(), &diagnostics);
            spec.spec.set_phrase_boundary(weight as FLOAT);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "melodic_continuity",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: INT| {
            spec.spec
                .record_call("melodic_continuity", call.call_position(), &diagnostics);
            spec.spec.set_melodic_continuity(weight as FLOAT);
            spec
        },
    );
    let diagnostics = ctx.lock().expect("lock script context").warnings.clone();
    engine.register_fn(
        "follow_tonal_context",
        move |call: NativeCallContext, mut spec: PopulationSpecHandle, weight: INT| {
            spec.spec
                .record_call("follow_tonal_context", call.call_position(), &diagnostics);
            spec.spec.set_follow_tonal_context(weight as FLOAT);
            spec
        },
    );
    // Tier 3: expert tuning
    register_population_spec_numeric_methods(
        engine,
        ctx.clone(),
        &[
            ("pulse_lock", PopulationSpec::set_pulse_lock),
            ("social", PopulationSpec::set_social),
        ],
    );
    register_population_spec_numeric_methods(
        engine,
        ctx.clone(),
        &[
            ("brightness", PopulationSpec::set_brightness),
            ("attack_noise", PopulationSpec::set_attack_noise),
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
        engine,
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
        engine,
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
        engine,
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
        engine,
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
        engine,
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
                diagnostics.lock().expect("lock script warnings").push(ScriptWarning { kind: "respawn_settle_placement", count: 1, message: "respawn_settle() requires consonance(), dissonance(), edge(), gap(), random(), or line()".to_string(), position: call_ctx.call_position()});
            }
            population_spec
        },
    );
}

pub(super) type PopulationSpecNumericSetter = fn(&mut PopulationSpec, f32);
type PopulationSpecPairNumericSetter = fn(&mut PopulationSpec, f32, f32);
type PopulationSpecNumericMethod = (&'static str, PopulationSpecNumericSetter);
type PopulationSpecPairNumericMethod = (&'static str, PopulationSpecPairNumericSetter);

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
        if name == "rest_on_habituation" && (!value.is_finite() || value < 0.0) {
            diagnostics.lock().expect("lock diagnostics").push(ScriptWarning {
                kind: "rest_on_habituation_invalid", count: 1, message: "rest_on_habituation() expects a finite nonnegative f32 weight; ignored".into(), position: call.call_position(),
            });
            setter(&mut population_spec.spec, 0.0);
        } else {
            setter(&mut population_spec.spec, value);
        }
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
