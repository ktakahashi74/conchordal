use super::*;

impl ScriptHost {
    pub fn create_engine(ctx: Arc<Mutex<ScriptContext>>) -> Engine {
        let mut engine = Engine::new();
        engine.on_print(|msg| println!("[rhai] {msg}"));

        engine.register_type_with_name::<PopulationSpecHandle>("PopulationSpec");
        engine.register_type_with_name::<PopulationHandle>("Population");
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
        engine.register_fn("variant", |parent: PopulationSpecHandle| parent);

        engine.register_fn("|", |left: Bus, right: Bus| left.set().combine(right.set()));
        engine.register_fn("|", |left: BusSet, right: Bus| left.combine(right.set()));
        engine.register_fn("|", |left: Bus, right: BusSet| left.set().combine(right));
        engine.register_fn("|", |left: BusSet, right: BusSet| left.combine(right));
        engine.register_fn(
            "send",
            |mut population_spec: PopulationSpecHandle, bus: Bus| {
                population_spec.spec.set_routing(bus.set().routing());
                population_spec
            },
        );
        engine.register_fn(
            "send",
            |mut population_spec: PopulationSpecHandle, bus: BusSet| {
                population_spec.spec.set_routing(bus.routing());
                population_spec
            },
        );

        register_population_spec_numeric_methods(
            &mut engine,
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
                ("pitch_smooth", PopulationSpec::set_pitch_smooth_tau),
            ],
        );
        register_population_spec_pair_numeric_methods(
            &mut engine,
            &[("avoid_neighbors", PopulationSpec::set_crowding)],
        );
        engine.register_fn(
            "avoid_neighbors",
            |mut population_spec: PopulationSpecHandle, strength: FLOAT| {
                population_spec
                    .spec
                    .set_crowding_strength_only(strength as f32);
                population_spec
            },
        );
        engine.register_fn(
            "avoid_neighbors",
            |mut population_spec: PopulationSpecHandle, strength: INT| {
                population_spec
                    .spec
                    .set_crowding_strength_only(strength as f32);
                population_spec
            },
        );
        engine.register_fn(
            "crowding_target",
            |mut population_spec: PopulationSpecHandle,
             same_population_visible: bool,
             other_population_visible: bool| {
                population_spec
                    .spec
                    .set_crowding_target(same_population_visible, other_population_visible);
                population_spec
            },
        );
        engine.register_fn(
            "leave_self_out",
            |mut population_spec: PopulationSpecHandle, enabled: bool| {
                population_spec.spec.set_leave_self_out(enabled);
                population_spec
            },
        );
        engine.register_fn(
            "leave_self_out_mode",
            |mut population_spec: PopulationSpecHandle, name: &str| {
                population_spec.spec.set_leave_self_out_mode(name);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            &[
                ("move_cost", PopulationSpec::set_move_cost_coeff),
                ("move_cost_exp", PopulationSpec::set_move_cost_exp),
                (
                    "proposal_interval",
                    PopulationSpec::set_proposal_interval_sec,
                ),
            ],
        );
        engine.register_fn(
            "global_peaks",
            |mut population_spec: PopulationSpecHandle, count: INT| {
                population_spec.spec.set_global_peaks(count, 0.0);
                population_spec
            },
        );
        engine.register_fn(
            "global_peaks",
            |mut population_spec: PopulationSpecHandle, count: INT, min_sep_cents: FLOAT| {
                population_spec
                    .spec
                    .set_global_peaks(count, min_sep_cents as f32);
                population_spec
            },
        );
        engine.register_fn(
            "global_peaks",
            |mut population_spec: PopulationSpecHandle, count: INT, min_sep_cents: INT| {
                population_spec
                    .spec
                    .set_global_peaks(count, min_sep_cents as f32);
                population_spec
            },
        );
        engine.register_fn(
            "ratio_candidates",
            |mut population_spec: PopulationSpecHandle, count: INT| {
                population_spec.spec.set_ratio_candidates(count);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            &[
                ("window_cents", PopulationSpec::set_window_cents),
                ("top_k", PopulationSpec::set_top_k),
                ("temperature", PopulationSpec::set_temperature),
                ("sigma_cents", PopulationSpec::set_sigma_cents),
                ("random_candidates", PopulationSpec::set_random_candidates),
            ],
        );
        engine.register_fn(
            "move_cost_time_scale",
            |mut population_spec: PopulationSpecHandle, name: &str| {
                population_spec.spec.set_move_cost_time_scale(name);
                population_spec
            },
        );
        engine.register_fn(
            "leave_self_out_harmonics",
            |mut population_spec: PopulationSpecHandle, value: INT| {
                population_spec.spec.set_leave_self_out_harmonics(value);
                population_spec
            },
        );
        engine.register_fn(
            "pitch_apply_mode",
            |mut population_spec: PopulationSpecHandle, name: &str| {
                population_spec.spec.set_pitch_apply_mode(name);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            &[("glide", PopulationSpec::set_pitch_glide_tau_sec)],
        );
        engine.register_fn(
            "seek_consonance",
            |mut population_spec: PopulationSpecHandle| {
                population_spec.spec.set_consonance_movement();
                population_spec
            },
        );
        engine.register_fn("anchor", |mut population_spec: PopulationSpecHandle| {
            population_spec.spec.set_anchor();
            population_spec
        });
        engine.register_fn(
            "pitch_core",
            |mut population_spec: PopulationSpecHandle, name: &str| {
                population_spec.spec.set_pitch_core(name);
                population_spec
            },
        );
        engine.register_fn(
            "brain",
            |call_ctx: NativeCallContext,
             mut population_spec: PopulationSpecHandle,
             name: &str|
             -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                population_spec
                    .spec
                    .set_brain(name, call_ctx.call_position())?;
                Ok(population_spec)
            },
        );
        engine.register_fn("sustain", |mut population_spec: PopulationSpecHandle| {
            population_spec.spec.set_phonation(PhonationKind::Sustain);
            population_spec
        });
        engine.register_fn("repeat", |mut population_spec: PopulationSpecHandle| {
            population_spec.spec.set_phonation(PhonationKind::Repeat);
            population_spec
        });
        // Tier-1 rhythm presets over the coupling continuum. No Hz argument: the
        // tempo region is the director's `temporal_basin`; a preset only sets how
        // the voice relates to the shared emergent beat (free .. locked).
        engine.register_fn("metric", |mut population_spec: PopulationSpecHandle| {
            population_spec.spec.set_metric();
            population_spec
        });
        engine.register_fn("entrained", |mut population_spec: PopulationSpecHandle| {
            population_spec.spec.set_entrained();
            population_spec
        });
        engine.register_fn("flow", |mut population_spec: PopulationSpecHandle| {
            population_spec.spec.set_flow();
            population_spec
        });
        engine.register_fn(
            "rhythm_role",
            |call_ctx: NativeCallContext,
             mut population_spec: PopulationSpecHandle,
             name: &str|
             -> Result<PopulationSpecHandle, Box<EvalAltResult>> {
                population_spec
                    .spec
                    .set_rhythm_role(name, call_ctx.call_position())?;
                Ok(population_spec)
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            &[
                ("entrainment", PopulationSpec::set_entrainment),
                ("microtiming", PopulationSpec::set_microtiming),
            ],
        );
        // Tier 2: explicit when/duration
        engine.register_fn("once", |mut population_spec: PopulationSpecHandle| {
            population_spec.spec.set_when_once();
            population_spec
        });
        register_population_spec_numeric_methods(
            &mut engine,
            &[("pulse", PopulationSpec::set_when_pulse)],
        );
        engine.register_fn(
            "while_alive",
            |mut population_spec: PopulationSpecHandle| {
                population_spec.spec.set_duration_while_alive();
                population_spec
            },
        );
        engine.register_fn(
            "phonate_when_viable",
            |mut population_spec: PopulationSpecHandle| {
                population_spec.spec.set_phonate_when_viable();
                population_spec
            },
        );
        engine.register_fn(
            "cycles",
            |mut population_spec: PopulationSpecHandle, n: INT| {
                population_spec.spec.set_duration_cycles(n.max(1) as u32);
                population_spec
            },
        );
        engine.register_fn(
            "adaptive_duration",
            |mut population_spec: PopulationSpecHandle| {
                population_spec.spec.set_adaptive_duration();
                population_spec
            },
        );
        // Tier 3: expert tuning
        register_population_spec_numeric_methods(
            &mut engine,
            &[
                ("pulse_lock", PopulationSpec::set_pulse_lock),
                ("social", PopulationSpec::set_social),
                ("shorten_on_drop", PopulationSpec::set_shorten_on_drop),
            ],
        );
        register_population_spec_pair_numeric_methods(
            &mut engine,
            &[
                ("duration_range", PopulationSpec::set_duration_range),
                ("duration_curve", PopulationSpec::set_duration_curve),
            ],
        );
        register_population_spec_numeric_methods(
            &mut engine,
            &[
                ("brightness", PopulationSpec::set_brightness),
                ("spread", PopulationSpec::set_spread),
                ("unison", PopulationSpec::set_unison),
            ],
        );
        engine.register_fn(
            "modes",
            |mut population_spec: PopulationSpecHandle, pattern: ModePattern| {
                population_spec.spec.set_modes(pattern);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
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
            &[(
                "consonance_viability",
                PopulationSpec::set_consonance_viability,
            )],
        );
        engine.register_fn(
            "viability_scope",
            |mut population_spec: PopulationSpecHandle, name: &str| {
                population_spec.spec.set_viability_scope(name);
                population_spec
            },
        );
        engine.register_fn(
            "selection_approx_loo",
            |mut population_spec: PopulationSpecHandle, enabled: bool| {
                population_spec.spec.set_selection_approx_loo(enabled);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            &[("dissonance_penalty", PopulationSpec::set_dissonance_penalty)],
        );
        engine.register_fn(
            "adsr",
            |mut population_spec: PopulationSpecHandle, a: FLOAT, d: FLOAT, s: FLOAT, r: FLOAT| {
                population_spec
                    .spec
                    .set_adsr(a as f32, d as f32, s as f32, r as f32);
                population_spec
            },
        );
        engine.register_fn(
            "rhythm_coupling_vitality",
            |mut population_spec: PopulationSpecHandle, lambda_v: FLOAT, v_floor: FLOAT| {
                population_spec
                    .spec
                    .set_rhythm_coupling_vitality(lambda_v as f32, v_floor as f32);
                population_spec
            },
        );
        engine.register_fn(
            "rhythm_reward",
            |mut population_spec: PopulationSpecHandle, rho_t: FLOAT, metric: &str| {
                population_spec.spec.set_rhythm_reward(rho_t as f32, metric);
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            &[("rhythm_freq", PopulationSpec::set_rhythm_freq)],
        );
        engine.register_fn(
            "respawn_random",
            |mut population_spec: PopulationSpecHandle| {
                population_spec.spec.set_respawn_random();
                population_spec
            },
        );
        engine.register_fn(
            "respawn_hereditary",
            |mut population_spec: PopulationSpecHandle, sigma_oct: FLOAT| {
                population_spec
                    .spec
                    .set_respawn_hereditary(sigma_oct as f32);
                population_spec
            },
        );
        engine.register_fn(
            "respawn_hereditary",
            |mut population_spec: PopulationSpecHandle, sigma_oct: INT| {
                population_spec
                    .spec
                    .set_respawn_hereditary(sigma_oct as f32);
                population_spec
            },
        );
        engine.register_fn(
            "respawn_consonance",
            |mut population_spec: PopulationSpecHandle| {
                population_spec.spec.set_respawn_consonance();
                population_spec
            },
        );
        register_population_spec_numeric_methods(
            &mut engine,
            &[
                ("respawn_capacity", PopulationSpec::set_respawn_capacity),
                (
                    "respawn_min_c_level",
                    PopulationSpec::set_respawn_min_c_level,
                ),
                (
                    "respawn_background_death_rate",
                    PopulationSpec::set_respawn_background_death_rate,
                ),
            ],
        );
        engine.register_fn(
            "respawn_settle",
            |mut population_spec: PopulationSpecHandle, placement: Placement| {
                if let Some(strategy) = placement.strategy() {
                    population_spec.spec.set_respawn_settle_strategy(strategy);
                } else {
                    warn!("respawn_settle() requires consonance(), dissonance(), edge(), gap(), random(), or line()");
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
                ctx.release_population(handle.id);
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
        engine.register_fn("power_modes", |beta: FLOAT| {
            ModePattern::power_modes(beta as f32)
        });
        engine.register_fn("stiff_string_modes", |stiffness: FLOAT| {
            ModePattern::stiff_string_modes(stiffness as f32)
        });
        engine.register_fn("custom_modes", |ratios: Array| {
            ModePattern::custom_modes(rhai_array_to_f32(ratios, "custom_modes"))
        });
        engine.register_fn("modal_table", |name: &str| {
            if let Some(pattern) = ModePattern::modal_table(name) {
                pattern
            } else {
                warn!(
                    "modal_table('{}') not found; falling back to harmonic_modes()",
                    name
                );
                ModePattern::harmonic_modes()
            }
        });
        engine.register_fn(
            "landscape_density_modes",
            ModePattern::landscape_density_modes,
        );
        engine.register_fn("landscape_peaks_modes", ModePattern::landscape_peaks_modes);
        engine.register_fn("count", |pattern: ModePattern, n: INT| {
            pattern.with_count((n as usize).max(1))
        });
        engine.register_fn(
            "range",
            |pattern: ModePattern, min_mul: FLOAT, max_mul: FLOAT| {
                if pattern.supports_range() {
                    pattern.with_range(min_mul as f32, max_mul as f32)
                } else {
                    warn!("range() is only supported for landscape_*_modes(); ignored");
                    pattern
                }
            },
        );
        engine.register_fn("spacing", |pattern: ModePattern, min_dist: FLOAT| {
            if pattern.supports_min_dist_erb() {
                pattern.with_min_dist_erb(min_dist as f32)
            } else {
                warn!("spacing() is only supported for landscape_*_modes(); ignored");
                pattern
            }
        });
        engine.register_fn("gamma", |pattern: ModePattern, gamma: FLOAT| {
            if pattern.supports_gamma() {
                pattern.with_gamma(gamma as f32)
            } else {
                warn!("gamma() is only supported for landscape_density_modes(); ignored");
                pattern
            }
        });
        engine.register_fn("jitter", |pattern: ModePattern, cents: FLOAT| {
            pattern.with_jitter_cents(cents as f32)
        });
        engine.register_fn("seed", |pattern: ModePattern, seed: INT| {
            if seed < 0 {
                warn!("seed() expects >= 0");
                pattern
            } else {
                pattern.with_seed(seed as u64)
            }
        });

        engine.register_fn("at", |freq: FLOAT| Placement::at(freq as f32));
        engine.register_fn("at", |freq: INT| Placement::at(freq as f32));
        // Field targets: consonance also takes a 1-arg root form; all field
        // targets and the geometric line take a 2-arg (lo, hi) range.
        engine.register_fn("consonance", |root: FLOAT| {
            Placement::consonance_root(root as f32)
        });
        engine.register_fn("consonance", |root: INT| {
            Placement::consonance_root(root as f32)
        });
        macro_rules! reg_range {
            ($name:expr, $ctor:path) => {
                engine.register_fn($name, |a: FLOAT, b: FLOAT| $ctor(a as f32, b as f32));
                engine.register_fn($name, |a: INT, b: FLOAT| $ctor(a as f32, b as f32));
                engine.register_fn($name, |a: FLOAT, b: INT| $ctor(a as f32, b as f32));
                engine.register_fn($name, |a: INT, b: INT| $ctor(a as f32, b as f32));
            };
        }
        reg_range!("consonance", Placement::consonance_range);
        reg_range!("dissonance", Placement::dissonance);
        reg_range!("edge", Placement::edge);
        reg_range!("gap", Placement::gap);
        reg_range!("random", Placement::random);
        reg_range!("line", Placement::line);
        engine.register_fn("peak", |placement: Placement| {
            placement.with_sampling(FieldSampling::Peak)
        });
        engine.register_fn("density", |placement: Placement| {
            placement.with_sampling(FieldSampling::Density)
        });
        engine.register_fn("tension", |placement: Placement, t: FLOAT| {
            placement.with_tension(t as f32)
        });
        engine.register_fn("tension", |placement: Placement, t: INT| {
            placement.with_tension(t as f32)
        });
        engine.register_fn("count", |placement: Placement, count: INT| {
            placement.with_count(count)
        });
        engine.register_fn(
            "range",
            |placement: Placement, min_mul: FLOAT, max_mul: FLOAT| {
                placement.with_range(min_mul as f32, max_mul as f32)
            },
        );
        engine.register_fn(
            "range",
            |placement: Placement, min_mul: INT, max_mul: FLOAT| {
                placement.with_range(min_mul as f32, max_mul as f32)
            },
        );
        engine.register_fn(
            "range",
            |placement: Placement, min_mul: FLOAT, max_mul: INT| {
                placement.with_range(min_mul as f32, max_mul as f32)
            },
        );
        engine.register_fn(
            "range",
            |placement: Placement, min_mul: INT, max_mul: INT| {
                placement.with_range(min_mul as f32, max_mul as f32)
            },
        );
        engine.register_fn("spacing", |placement: Placement, spacing: FLOAT| {
            placement.with_spacing(spacing as f32)
        });
        engine.register_fn("spacing", |placement: Placement, spacing: INT| {
            placement.with_spacing(spacing as f32)
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
                (
                    "pitch_smooth",
                    PopulationSpec::set_pitch_smooth_tau,
                    patch_pitch_smooth_tau,
                ),
            ],
        );
        register_population_crowding_overloads(&mut engine, ctx.clone(), "avoid_neighbors");
        let ctx_for_population_crowding_target = ctx.clone();
        engine.register_fn(
            "crowding_target",
            move |handle: PopulationHandle,
                  same_population_visible: bool,
                  other_population_visible: bool|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_crowding_target
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!(
                        "crowding_target ignored for unknown population {}",
                        handle.id
                    );
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
                    _ => ctx.warn_population_inactive(handle.id, "crowding_target"),
                }
                Ok(handle)
            },
        );
        let ctx_for_population_leave_self_out = ctx.clone();
        engine.register_fn(
            "leave_self_out",
            move |handle: PopulationHandle,
                  enabled: bool|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_leave_self_out
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!(
                        "leave_self_out ignored for unknown population {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                match population.status {
                    PopulationStatus::Live => {
                        population.spec.set_leave_self_out(enabled);
                        population.pending_patch.leave_self_out = Some(enabled);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "leave_self_out"),
                }
                Ok(handle)
            },
        );
        let ctx_for_population_leave_self_out_mode = ctx.clone();
        engine.register_fn(
            "leave_self_out_mode",
            move |handle: PopulationHandle,
                  name: &str|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_leave_self_out_mode
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!(
                        "leave_self_out_mode ignored for unknown population {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                let mode = parse_leave_self_out_mode_name(
                    population.spec.control.pitch.leave_self_out_mode,
                    name,
                );
                match population.status {
                    PopulationStatus::Live => {
                        population.spec.control.pitch.set_leave_self_out_mode(mode);
                        patch_leave_self_out_mode(&mut population.pending_patch, mode);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "leave_self_out_mode"),
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
        engine.register_fn(
            "global_peaks",
            move |handle: PopulationHandle,
                  count: INT|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_global_peaks
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown population {}", handle.id);
                    return Ok(handle);
                };
                match population.status {
                    PopulationStatus::Live => {
                        population.spec.set_global_peaks(count, 0.0);
                        population.pending_patch.global_peak_count = Some(count);
                        population.pending_patch.global_peak_min_sep_cents = Some(0.0);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_population_global_peaks_sep = ctx.clone();
        engine.register_fn(
            "global_peaks",
            move |handle: PopulationHandle,
                  count: INT,
                  min_sep_cents: FLOAT|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_global_peaks_sep
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown population {}", handle.id);
                    return Ok(handle);
                };
                let min_sep = min_sep_cents as f32;
                match population.status {
                    PopulationStatus::Live => {
                        population.spec.set_global_peaks(count, min_sep);
                        population.pending_patch.global_peak_count = Some(count);
                        population.pending_patch.global_peak_min_sep_cents = Some(min_sep);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_population_global_peaks_sep_int = ctx.clone();
        engine.register_fn(
            "global_peaks",
            move |handle: PopulationHandle,
                  count: INT,
                  min_sep_cents: INT|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_global_peaks_sep_int
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown population {}", handle.id);
                    return Ok(handle);
                };
                let min_sep = min_sep_cents as f32;
                match population.status {
                    PopulationStatus::Live => {
                        population.spec.set_global_peaks(count, min_sep);
                        population.pending_patch.global_peak_count = Some(count);
                        population.pending_patch.global_peak_min_sep_cents = Some(min_sep);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_population_ratio_candidates = ctx.clone();
        engine.register_fn(
            "ratio_candidates",
            move |handle: PopulationHandle,
                  count: INT|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_ratio_candidates
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!(
                        "ratio_candidates ignored for unknown population {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                match population.status {
                    PopulationStatus::Live => {
                        population.spec.set_ratio_candidates(count);
                        population.pending_patch.ratio_candidate_count = Some(count);
                        population.pending_patch.use_ratio_candidates = Some(count > 0);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "ratio_candidates"),
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
                ("top_k", PopulationSpec::set_top_k, patch_top_k),
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
                (
                    "random_candidates",
                    PopulationSpec::set_random_candidates,
                    patch_random_candidates,
                ),
            ],
        );
        let ctx_for_population_move_cost_time_scale = ctx.clone();
        engine.register_fn(
            "move_cost_time_scale",
            move |handle: PopulationHandle,
                  name: &str|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_move_cost_time_scale
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!(
                        "move_cost_time_scale ignored for unknown population {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                let lowered = name.trim().to_ascii_lowercase();
                let value = match lowered.as_str() {
                    "legacy" | "integration" | "integration_window" => {
                        MoveCostTimeScale::LegacyIntegrationWindow
                    }
                    "proposal" | "proposal_interval" => MoveCostTimeScale::ProposalInterval,
                    _ => {
                        ctx.warn_population_inactive(handle.id, "move_cost_time_scale");
                        return Ok(handle);
                    }
                };
                match population.status {
                    PopulationStatus::Live => {
                        population
                            .spec
                            .control
                            .pitch
                            .set_move_cost_time_scale(value);
                        population.pending_patch.move_cost_time_scale = Some(value);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "move_cost_time_scale"),
                }
                Ok(handle)
            },
        );
        let ctx_for_population_loo_harmonics = ctx.clone();
        engine.register_fn(
            "leave_self_out_harmonics",
            move |handle: PopulationHandle,
                  value: INT|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_loo_harmonics
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!(
                        "leave_self_out_harmonics ignored for unknown population {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                match population.status {
                    PopulationStatus::Live => {
                        population.spec.set_leave_self_out_harmonics(value);
                        population.pending_patch.leave_self_out_harmonics = Some(value);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "leave_self_out_harmonics"),
                }
                Ok(handle)
            },
        );
        let ctx_for_population_pitch_apply_mode = ctx.clone();
        engine.register_fn(
            "pitch_apply_mode",
            move |handle: PopulationHandle,
                  name: &str|
                  -> Result<PopulationHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_population_pitch_apply_mode
                    .lock()
                    .expect("lock script context");
                let Some(population) = ctx.populations.get_mut(&handle.id) else {
                    warn!(
                        "pitch_apply_mode ignored for unknown population {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                let lowered = name.trim().to_ascii_lowercase();
                let mode = match lowered.as_str() {
                    "gate_snap" | "gatesnap" | "snap" => PitchApplyMode::GateSnap,
                    "glide" | "gliss" | "glissando" => PitchApplyMode::Glide,
                    _ => {
                        ctx.warn_population_inactive(handle.id, "pitch_apply_mode");
                        return Ok(handle);
                    }
                };
                match population.status {
                    PopulationStatus::Live => {
                        population.spec.set_pitch_apply_mode_resolved(mode);
                        population.pending_patch.pitch_apply_mode = Some(mode);
                    }
                    _ => ctx.warn_population_inactive(handle.id, "pitch_apply_mode"),
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
                ("unison", PopulationSpec::set_unison, patch_timbre_unison),
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
            move |_call_ctx: NativeCallContext, name: &str| {
                let mut ctx = ctx_for_set_pitch_objective
                    .lock()
                    .expect("lock script context");
                let lowered = name.trim().to_ascii_lowercase();
                let mode = match lowered.as_str() {
                    "consonance" | "positive" | "pos" => PitchObjectiveMode::Consonance,
                    "negative_consonance" | "negative" | "neg" | "dissonance" => {
                        PitchObjectiveMode::NegativeConsonance
                    }
                    other => {
                        warn!(
                            "set_pitch_objective() expects 'consonance' or 'negative_consonance', got '{}'",
                            other
                        );
                        return;
                    }
                };
                let cursor = ctx.cursor;
                let update = crate::core::landscape::LandscapeUpdate {
                    pitch_objective_mode: Some(mode),
                    ..crate::core::landscape::LandscapeUpdate::default()
                };
                ctx.push_event(cursor, vec![Action::SetHarmonicityParams { update }]);
            },
        );

        let ctx_for_set_global_coupling = ctx.clone();
        engine.register_fn(
            "set_global_coupling",
            move |_call_ctx: NativeCallContext, value: FLOAT| {
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
            "set_control_update_mode",
            move |_call_ctx: NativeCallContext, name: &str| {
                let mut ctx = ctx_for_set_control_update_mode
                    .lock()
                    .expect("lock script context");
                let lowered = name.trim().to_ascii_lowercase();
                let mode = match lowered.as_str() {
                    "snapshot_phased" | "snapshot" => ControlUpdateMode::SnapshotPhased,
                    "sequential_rotating" | "sequential" => {
                        ControlUpdateMode::SequentialRotating
                    }
                    other => {
                        warn!(
                            "set_control_update_mode() expects 'snapshot_phased' or 'sequential_rotating', got '{}'",
                            other
                        );
                        return;
                    }
                };
                ctx.scenario.control_update_mode = mode;
            },
        );

        let ctx_for_set_scaffold_off = ctx.clone();
        engine.register_fn("set_scaffold_off", move |_call_ctx: NativeCallContext| {
            let mut ctx = ctx_for_set_scaffold_off
                .lock()
                .expect("lock script context");
            ctx.scenario.scaffold = ScaffoldConfig::Off;
        });
        let ctx_for_set_scaffold_shared = ctx.clone();
        engine.register_fn(
            "set_scaffold_shared",
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
            "set_scaffold_shared",
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
            "set_scaffold_scrambled",
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
            "set_scaffold_scrambled",
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

        let ctx_for_set_roughness_k = ctx.clone();
        engine.register_fn(
            "set_roughness_k",
            move |_call_ctx: NativeCallContext, value: FLOAT| {
                let mut ctx = ctx_for_set_roughness_k.lock().expect("lock script context");
                let cursor = ctx.cursor;
                ctx.push_event(
                    cursor,
                    vec![Action::SetRoughnessTolerance {
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
        let src = fs::read_to_string(path)
            .map_err(|err| ScriptError::new(format!("read script {path}: {err}"), None))?;
        let ctx = Arc::new(Mutex::new(match seed {
            Some(seed) => ScriptContext::with_seed(seed),
            None => ScriptContext::default(),
        }));
        let engine = ScriptHost::create_engine(ctx.clone());

        if let Err(e) = engine.eval::<()>(&src) {
            println!("Debug script error: {:?}", e);
            return Err(ScriptError::from_eval(
                e,
                Some(&format!("execute script {path}")),
            ));
        }

        let mut ctx_out = ctx.lock().expect("lock script context");
        ctx_out.finish();
        Ok(ctx_out.scenario.clone())
    }
}
