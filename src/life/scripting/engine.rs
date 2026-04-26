use super::*;

impl ScriptHost {
    pub(super) fn create_engine(ctx: Arc<Mutex<ScriptContext>>) -> Engine {
        let mut engine = Engine::new();
        engine.on_print(|msg| println!("[rhai] {msg}"));

        engine.register_type_with_name::<SpeciesHandle>("SpeciesHandle");
        engine.register_type_with_name::<GroupHandle>("GroupHandle");
        engine.register_type_with_name::<SpawnStrategy>("SpawnStrategy");
        engine.register_type_with_name::<ModePattern>("ModePattern");

        let mut presets = rhai::Module::new();
        presets.set_var(
            "sine",
            SpeciesHandle {
                spec: SpeciesSpec::preset(BodyMethod::Sine),
            },
        );
        presets.set_var(
            "harmonic",
            SpeciesHandle {
                spec: SpeciesSpec::preset(BodyMethod::Harmonic),
            },
        );
        presets.set_var(
            "saw",
            SpeciesHandle {
                spec: {
                    let mut spec = SpeciesSpec::preset(BodyMethod::Harmonic);
                    spec.control.body.timbre.brightness = 0.85;
                    spec
                },
            },
        );
        presets.set_var(
            "square",
            SpeciesHandle {
                spec: {
                    let mut spec = SpeciesSpec::preset(BodyMethod::Harmonic);
                    spec.control.body.timbre.brightness = 0.65;
                    spec
                },
            },
        );
        presets.set_var(
            "noise",
            SpeciesHandle {
                spec: {
                    let mut spec = SpeciesSpec::preset(BodyMethod::Harmonic);
                    spec.control.body.timbre.brightness = 1.0;
                    spec.control.body.timbre.motion = 1.0;
                    spec
                },
            },
        );
        presets.set_var(
            "modal",
            SpeciesHandle {
                spec: SpeciesSpec::preset(BodyMethod::Modal),
            },
        );
        engine.register_global_module(presets.into());

        engine.register_fn("derive", |parent: SpeciesHandle| parent);

        register_species_numeric_overloads(&mut engine, "amp", SpeciesSpec::set_amp);
        register_species_numeric_overloads(&mut engine, "freq", SpeciesSpec::set_freq);
        register_species_numeric_overloads(
            &mut engine,
            "landscape_weight",
            SpeciesSpec::set_landscape_weight,
        );
        register_species_numeric_overloads(
            &mut engine,
            "neighbor_step_cents",
            SpeciesSpec::set_neighbor_step_cents,
        );
        register_species_numeric_overloads(
            &mut engine,
            "tessitura_gravity",
            SpeciesSpec::set_tessitura_gravity,
        );
        register_species_numeric_overloads(
            &mut engine,
            "sustain_drive",
            SpeciesSpec::set_continuous_drive,
        );
        register_species_numeric_overloads(
            &mut engine,
            "pitch_smooth",
            SpeciesSpec::set_pitch_smooth_tau,
        );
        register_species_numeric_overloads(
            &mut engine,
            "exploration",
            SpeciesSpec::set_exploration,
        );
        register_species_numeric_overloads(
            &mut engine,
            "persistence",
            SpeciesSpec::set_persistence,
        );
        register_species_pair_numeric_overloads(&mut engine, "crowding", SpeciesSpec::set_crowding);
        engine.register_fn("crowding", |mut species: SpeciesHandle, strength: FLOAT| {
            species.spec.set_crowding_auto_sigma(strength as f32);
            species
        });
        engine.register_fn("crowding", |mut species: SpeciesHandle, strength: INT| {
            species.spec.set_crowding_auto_sigma(strength as f32);
            species
        });
        engine.register_fn(
            "crowding_target",
            |mut species: SpeciesHandle, same_group_visible: bool, other_group_visible: bool| {
                species
                    .spec
                    .set_crowding_target(same_group_visible, other_group_visible);
                species
            },
        );
        engine.register_fn(
            "leave_self_out",
            |mut species: SpeciesHandle, enabled: bool| {
                species.spec.set_leave_self_out(enabled);
                species
            },
        );
        engine.register_fn(
            "leave_self_out_mode",
            |mut species: SpeciesHandle, name: &str| {
                species.spec.set_leave_self_out_mode(name);
                species
            },
        );
        register_species_numeric_overloads(
            &mut engine,
            "anneal_temp",
            SpeciesSpec::set_anneal_temp,
        );
        register_species_numeric_overloads(
            &mut engine,
            "move_cost",
            SpeciesSpec::set_move_cost_coeff,
        );
        register_species_numeric_overloads(
            &mut engine,
            "move_cost_exp",
            SpeciesSpec::set_move_cost_exp,
        );
        register_species_numeric_overloads(
            &mut engine,
            "improvement_threshold",
            SpeciesSpec::set_improvement_threshold,
        );
        register_species_numeric_overloads(
            &mut engine,
            "proposal_interval",
            SpeciesSpec::set_proposal_interval_sec,
        );
        engine.register_fn("global_peaks", |mut species: SpeciesHandle, count: INT| {
            species.spec.set_global_peaks(count, 0.0);
            species
        });
        engine.register_fn(
            "global_peaks",
            |mut species: SpeciesHandle, count: INT, min_sep_cents: FLOAT| {
                species.spec.set_global_peaks(count, min_sep_cents as f32);
                species
            },
        );
        engine.register_fn(
            "global_peaks",
            |mut species: SpeciesHandle, count: INT, min_sep_cents: INT| {
                species.spec.set_global_peaks(count, min_sep_cents as f32);
                species
            },
        );
        engine.register_fn(
            "ratio_candidates",
            |mut species: SpeciesHandle, count: INT| {
                species.spec.set_ratio_candidates(count);
                species
            },
        );
        register_species_numeric_overloads(
            &mut engine,
            "window_cents",
            SpeciesSpec::set_window_cents,
        );
        register_species_numeric_overloads(&mut engine, "top_k", SpeciesSpec::set_top_k);
        register_species_numeric_overloads(
            &mut engine,
            "temperature",
            SpeciesSpec::set_temperature,
        );
        register_species_numeric_overloads(
            &mut engine,
            "sigma_cents",
            SpeciesSpec::set_sigma_cents,
        );
        register_species_numeric_overloads(
            &mut engine,
            "random_candidates",
            SpeciesSpec::set_random_candidates,
        );
        engine.register_fn(
            "move_cost_time_scale",
            |mut species: SpeciesHandle, name: &str| {
                species.spec.set_move_cost_time_scale(name);
                species
            },
        );
        engine.register_fn(
            "leave_self_out_harmonics",
            |mut species: SpeciesHandle, value: INT| {
                species.spec.set_leave_self_out_harmonics(value);
                species
            },
        );
        engine.register_fn(
            "pitch_apply_mode",
            |mut species: SpeciesHandle, name: &str| {
                species.spec.set_pitch_apply_mode(name);
                species
            },
        );
        register_species_numeric_overloads(
            &mut engine,
            "pitch_glide",
            SpeciesSpec::set_pitch_glide_tau_sec,
        );
        engine.register_fn("pitch_mode", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_pitch_mode(name);
            species
        });
        engine.register_fn("pitch_core", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_pitch_core(name);
            species
        });
        engine.register_fn("brain", |mut species: SpeciesHandle, name: &str| {
            species.spec.set_brain(name);
            species
        });
        engine.register_fn("sustain", |mut species: SpeciesHandle| {
            species.spec.set_phonation(PhonationKind::Sustain);
            species
        });
        engine.register_fn("repeat", |mut species: SpeciesHandle| {
            species.spec.set_phonation(PhonationKind::Repeat);
            species
        });
        // Routing: .mute() bypasses listener output, .unperceived() bypasses voice perception.
        engine.register_fn("mute", |mut species: SpeciesHandle| {
            species.spec.control.body.routing.to_listener = false;
            species
        });
        engine.register_fn("unperceived", |mut species: SpeciesHandle| {
            species.spec.control.body.routing.to_voices = false;
            species
        });
        // Tier 2: explicit when/duration
        engine.register_fn("once", |mut species: SpeciesHandle| {
            species.spec.set_when_once();
            species
        });
        engine.register_fn("pulse", |mut species: SpeciesHandle, rate: FLOAT| {
            species.spec.set_when_pulse(rate as f32);
            species
        });
        engine.register_fn("while_alive", |mut species: SpeciesHandle| {
            species.spec.set_duration_while_alive();
            species
        });
        engine.register_fn("gates", |mut species: SpeciesHandle, n: INT| {
            species.spec.set_duration_gates(n.max(1) as u32);
            species
        });
        engine.register_fn("field", |mut species: SpeciesHandle| {
            species.spec.set_duration_field();
            species
        });
        // Tier 3: expert tuning
        engine.register_fn("sync", |mut species: SpeciesHandle, depth: FLOAT| {
            species.spec.set_sync(depth as f32);
            species
        });
        engine.register_fn("social", |mut species: SpeciesHandle, coupling: FLOAT| {
            species.spec.set_social(coupling as f32);
            species
        });
        engine.register_fn(
            "field_window",
            |mut species: SpeciesHandle, min: FLOAT, max: FLOAT| {
                species.spec.set_field_window(min as f32, max as f32);
                species
            },
        );
        engine.register_fn(
            "field_curve",
            |mut species: SpeciesHandle, k: FLOAT, x0: FLOAT| {
                species.spec.set_field_curve(k as f32, x0 as f32);
                species
            },
        );
        engine.register_fn("field_drop", |mut species: SpeciesHandle, gain: FLOAT| {
            species.spec.set_field_drop(gain as f32);
            species
        });
        register_species_numeric_overloads(&mut engine, "brightness", SpeciesSpec::set_brightness);
        register_species_numeric_overloads(&mut engine, "spread", SpeciesSpec::set_spread);
        register_species_numeric_overloads(&mut engine, "unison", SpeciesSpec::set_unison);
        engine.register_fn(
            "modes",
            |mut species: SpeciesHandle, pattern: ModePattern| {
                species.spec.set_modes(pattern);
                species
            },
        );
        engine.register_fn("metabolism", |mut species: SpeciesHandle, rate: FLOAT| {
            species.spec.set_metabolism(rate as f32);
            species
        });
        register_species_numeric_overloads(
            &mut engine,
            "initial_energy",
            SpeciesSpec::set_initial_energy,
        );
        register_species_numeric_overloads(
            &mut engine,
            "recharge_rate",
            SpeciesSpec::set_recharge_rate,
        );
        register_species_numeric_overloads(
            &mut engine,
            "action_cost",
            SpeciesSpec::set_action_cost,
        );
        register_species_numeric_overloads(
            &mut engine,
            "continuous_recharge_rate",
            SpeciesSpec::set_continuous_recharge_rate,
        );
        register_species_pair_numeric_overloads(
            &mut engine,
            "survival_signal",
            SpeciesSpec::set_survival_signal,
        );
        engine.register_fn(
            "selection_approx_loo",
            |mut species: SpeciesHandle, enabled: bool| {
                species.spec.set_selection_approx_loo(enabled);
                species
            },
        );
        register_species_numeric_overloads(
            &mut engine,
            "dissonance_cost",
            SpeciesSpec::set_dissonance_cost,
        );
        register_species_numeric_overloads(&mut engine, "energy_cap", SpeciesSpec::set_energy_cap);
        engine.register_fn(
            "adsr",
            |mut species: SpeciesHandle, a: FLOAT, d: FLOAT, s: FLOAT, r: FLOAT| {
                species
                    .spec
                    .set_adsr(a as f32, d as f32, s as f32, r as f32);
                species
            },
        );
        engine.register_fn(
            "rhythm_coupling",
            |mut species: SpeciesHandle, mode: &str| {
                species.spec.set_rhythm_coupling(mode);
                species
            },
        );
        engine.register_fn(
            "rhythm_coupling_vitality",
            |mut species: SpeciesHandle, lambda_v: FLOAT, v_floor: FLOAT| {
                species
                    .spec
                    .set_rhythm_coupling_vitality(lambda_v as f32, v_floor as f32);
                species
            },
        );
        engine.register_fn(
            "rhythm_reward",
            |mut species: SpeciesHandle, rho_t: FLOAT, metric: &str| {
                species.spec.set_rhythm_reward(rho_t as f32, metric);
                species
            },
        );
        register_species_numeric_overloads(
            &mut engine,
            "rhythm_freq",
            SpeciesSpec::set_rhythm_freq,
        );
        register_species_numeric_overloads(
            &mut engine,
            "rhythm_sensitivity",
            SpeciesSpec::set_rhythm_sensitivity,
        );
        register_species_numeric_overloads(&mut engine, "k_omega", SpeciesSpec::set_k_omega);
        register_species_numeric_overloads(&mut engine, "base_sigma", SpeciesSpec::set_base_sigma);
        register_species_pair_numeric_overloads(
            &mut engine,
            "gate_thresholds",
            |species, env_open, mag| species.set_gate_thresholds(env_open, mag, 0.2, 0.9),
        );
        engine.register_fn(
            "gate_thresholds",
            |mut species: SpeciesHandle, env_open: FLOAT, mag: FLOAT, alpha: FLOAT, beta: FLOAT| {
                species.spec.set_gate_thresholds(
                    env_open as f32,
                    mag as f32,
                    alpha as f32,
                    beta as f32,
                );
                species
            },
        );
        engine.register_fn("respawn_random", |mut species: SpeciesHandle| {
            species.spec.set_respawn_random();
            species
        });
        engine.register_fn(
            "respawn_hereditary",
            |mut species: SpeciesHandle, sigma_oct: FLOAT| {
                species.spec.set_respawn_hereditary(sigma_oct as f32);
                species
            },
        );
        engine.register_fn(
            "respawn_hereditary",
            |mut species: SpeciesHandle, sigma_oct: INT| {
                species.spec.set_respawn_hereditary(sigma_oct as f32);
                species
            },
        );
        engine.register_fn("respawn_peak_bias", |mut species: SpeciesHandle| {
            species.spec.set_respawn_peak_bias();
            species
        });
        register_species_numeric_overloads(
            &mut engine,
            "respawn_capacity",
            SpeciesSpec::set_respawn_capacity,
        );
        register_species_numeric_overloads(
            &mut engine,
            "respawn_min_c_level",
            SpeciesSpec::set_respawn_min_c_level,
        );
        register_species_numeric_overloads(
            &mut engine,
            "respawn_background_death_rate",
            SpeciesSpec::set_respawn_background_death_rate,
        );
        engine.register_fn(
            "respawn_settle",
            |mut species: SpeciesHandle, strategy: SpawnStrategy| {
                species.spec.set_respawn_settle_strategy(strategy);
                species
            },
        );

        let ctx_for_create = ctx.clone();
        engine.register_fn(
            "create",
            move |call_ctx: NativeCallContext, species: SpeciesHandle, count: INT| {
                let mut ctx = ctx_for_create.lock().expect("lock script context");
                ctx.create_group(species, count, call_ctx.call_position())
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
            move |_call_ctx: NativeCallContext, handle: GroupHandle| {
                let mut ctx = ctx_for_release.lock().expect("lock script context");
                ctx.release_group(handle.id);
            },
        );

        let ctx_for_scene = ctx.clone();
        engine.register_fn(
            "scene",
            move |call_ctx: NativeCallContext, name: &str, callback: FnPtr| {
                {
                    let mut ctx = ctx_for_scene.lock().expect("lock script context");
                    ctx.push_scene_marker(name);
                    ctx.push_scope();
                }
                let result = callback.call_within_context::<Dynamic>(&call_ctx, ());
                let mut ctx = ctx_for_scene.lock().expect("lock script context");
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
        engine.register_fn("min_dist", |pattern: ModePattern, min_dist: FLOAT| {
            if pattern.supports_min_dist_erb() {
                pattern.with_min_dist_erb(min_dist as f32)
            } else {
                warn!("min_dist() is only supported for landscape_*_modes(); ignored");
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

        engine.register_fn("consonance", |root_freq: FLOAT| SpawnStrategy::Consonance {
            root_freq: root_freq as f32,
            min_mul: 1.0,
            max_mul: 4.0,
            min_dist_erb: 1.0,
        });
        engine.register_fn(
            "range",
            |strategy: SpawnStrategy, min_mul: FLOAT, max_mul: FLOAT| match strategy {
                SpawnStrategy::Consonance {
                    root_freq,
                    min_dist_erb,
                    ..
                } => SpawnStrategy::Consonance {
                    root_freq,
                    min_mul: min_mul as f32,
                    max_mul: max_mul as f32,
                    min_dist_erb,
                },
                other => {
                    warn!("range() ignored for non-consonance strategy");
                    other
                }
            },
        );
        engine.register_fn(
            "min_dist",
            |strategy: SpawnStrategy, min_dist: FLOAT| match strategy {
                SpawnStrategy::Consonance {
                    root_freq,
                    min_mul,
                    max_mul,
                    ..
                } => SpawnStrategy::Consonance {
                    root_freq,
                    min_mul,
                    max_mul,
                    min_dist_erb: min_dist as f32,
                },
                SpawnStrategy::ConsonanceDensity {
                    min_freq, max_freq, ..
                } => SpawnStrategy::ConsonanceDensity {
                    min_freq,
                    max_freq,
                    min_dist_erb: min_dist as f32,
                },
                other => {
                    warn!("min_dist() ignored for non-consonance strategy");
                    other
                }
            },
        );
        engine.register_fn(
            "consonance_density_pmf",
            |min_freq: FLOAT, max_freq: FLOAT| SpawnStrategy::ConsonanceDensity {
                min_freq: min_freq as f32,
                max_freq: max_freq as f32,
                min_dist_erb: 1.0,
            },
        );
        engine.register_fn("random_log", |min_freq: FLOAT, max_freq: FLOAT| {
            SpawnStrategy::RandomLog {
                min_freq: min_freq as f32,
                max_freq: max_freq as f32,
            }
        });
        engine.register_fn("linear", |start: FLOAT, end: FLOAT| SpawnStrategy::Linear {
            start_freq: start as f32,
            end_freq: end as f32,
        });
        engine.register_fn(
            "reject_targets",
            |strategy: SpawnStrategy,
             anchor_hz: FLOAT,
             targets_st: Array,
             exclusion_st: FLOAT,
             max_tries: INT| {
                SpawnStrategy::RejectTargets {
                    base: Box::new(strategy),
                    anchor_hz: anchor_hz as f32,
                    targets_st: rhai_array_to_f32(targets_st, "reject_targets"),
                    exclusion_st: exclusion_st as f32,
                    max_tries: max_tries.max(1) as usize,
                }
            },
        );
        engine.register_fn(
            "reject_targets",
            |strategy: SpawnStrategy,
             anchor_hz: INT,
             targets_st: Array,
             exclusion_st: FLOAT,
             max_tries: INT| {
                SpawnStrategy::RejectTargets {
                    base: Box::new(strategy),
                    anchor_hz: anchor_hz as f32,
                    targets_st: rhai_array_to_f32(targets_st, "reject_targets"),
                    exclusion_st: exclusion_st as f32,
                    max_tries: max_tries.max(1) as usize,
                }
            },
        );
        engine.register_fn(
            "reject_targets",
            |strategy: SpawnStrategy,
             anchor_hz: FLOAT,
             targets_st: Array,
             exclusion_st: INT,
             max_tries: INT| {
                SpawnStrategy::RejectTargets {
                    base: Box::new(strategy),
                    anchor_hz: anchor_hz as f32,
                    targets_st: rhai_array_to_f32(targets_st, "reject_targets"),
                    exclusion_st: exclusion_st as f32,
                    max_tries: max_tries.max(1) as usize,
                }
            },
        );
        engine.register_fn(
            "reject_targets",
            |strategy: SpawnStrategy,
             anchor_hz: INT,
             targets_st: Array,
             exclusion_st: INT,
             max_tries: INT| {
                SpawnStrategy::RejectTargets {
                    base: Box::new(strategy),
                    anchor_hz: anchor_hz as f32,
                    targets_st: rhai_array_to_f32(targets_st, "reject_targets"),
                    exclusion_st: exclusion_st as f32,
                    max_tries: max_tries.max(1) as usize,
                }
            },
        );

        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "amp",
            SpeciesSpec::set_amp,
            patch_amp,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "freq",
            SpeciesSpec::set_freq,
            patch_freq,
            Some(draft_clear_strategy),
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "landscape_weight",
            SpeciesSpec::set_landscape_weight,
            patch_landscape_weight,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "neighbor_step_cents",
            SpeciesSpec::set_neighbor_step_cents,
            patch_neighbor_step_cents,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "tessitura_gravity",
            SpeciesSpec::set_tessitura_gravity,
            patch_tessitura_gravity,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "sustain_drive",
            SpeciesSpec::set_continuous_drive,
            patch_continuous_drive,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "pitch_smooth",
            SpeciesSpec::set_pitch_smooth_tau,
            patch_pitch_smooth_tau,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "exploration",
            SpeciesSpec::set_exploration,
            patch_exploration,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "persistence",
            SpeciesSpec::set_persistence,
            patch_persistence,
            None,
        );
        register_group_crowding_overloads(&mut engine, ctx.clone());
        let ctx_for_group_crowding_target = ctx.clone();
        engine.register_fn(
            "crowding_target",
            move |handle: GroupHandle,
                  same_group_visible: bool,
                  other_group_visible: bool|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_crowding_target
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("crowding_target ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.crowding_target_same = same_group_visible;
                        group.crowding_target_other = other_group_visible;
                        group
                            .spec
                            .set_crowding_target(same_group_visible, other_group_visible);
                    }
                    GroupStatus::Live => {
                        group.crowding_target_same = same_group_visible;
                        group.crowding_target_other = other_group_visible;
                        group
                            .spec
                            .set_crowding_target(same_group_visible, other_group_visible);
                        group.pending_crowding_target =
                            Some((same_group_visible, other_group_visible));
                    }
                    _ => ctx.warn_live_builder(handle.id, "crowding_target"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_leave_self_out = ctx.clone();
        engine.register_fn(
            "leave_self_out",
            move |handle: GroupHandle, enabled: bool| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_leave_self_out
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("leave_self_out ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_leave_self_out(enabled),
                    GroupStatus::Live => {
                        group.spec.set_leave_self_out(enabled);
                        group.pending_patch.leave_self_out = Some(enabled);
                    }
                    _ => ctx.warn_live_builder(handle.id, "leave_self_out"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_leave_self_out_mode = ctx.clone();
        engine.register_fn(
            "leave_self_out_mode",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_leave_self_out_mode
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "leave_self_out_mode ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                let mode = parse_leave_self_out_mode_name(
                    group.spec.control.pitch.leave_self_out_mode,
                    name,
                );
                match group.status {
                    GroupStatus::Draft => group.spec.control.set_leave_self_out_mode(mode),
                    GroupStatus::Live => {
                        group.spec.control.set_leave_self_out_mode(mode);
                        patch_leave_self_out_mode(&mut group.pending_patch, mode);
                    }
                    _ => ctx.warn_live_builder(handle.id, "leave_self_out_mode"),
                }
                Ok(handle)
            },
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "anneal_temp",
            SpeciesSpec::set_anneal_temp,
            patch_anneal_temp,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "move_cost",
            SpeciesSpec::set_move_cost_coeff,
            patch_move_cost_coeff,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "move_cost_exp",
            SpeciesSpec::set_move_cost_exp,
            patch_move_cost_exp,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "improvement_threshold",
            SpeciesSpec::set_improvement_threshold,
            patch_improvement_threshold,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "proposal_interval",
            SpeciesSpec::set_proposal_interval_sec,
            patch_proposal_interval,
            None,
        );
        let ctx_for_group_global_peaks = ctx.clone();
        engine.register_fn(
            "global_peaks",
            move |handle: GroupHandle, count: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_global_peaks
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_global_peaks(count, 0.0),
                    GroupStatus::Live => {
                        group.spec.set_global_peaks(count, 0.0);
                        group.pending_patch.global_peak_count = Some(count);
                        group.pending_patch.global_peak_min_sep_cents = Some(0.0);
                    }
                    _ => ctx.warn_live_builder(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_global_peaks_sep = ctx.clone();
        engine.register_fn(
            "global_peaks",
            move |handle: GroupHandle,
                  count: INT,
                  min_sep_cents: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_global_peaks_sep
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let min_sep = min_sep_cents as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_global_peaks(count, min_sep),
                    GroupStatus::Live => {
                        group.spec.set_global_peaks(count, min_sep);
                        group.pending_patch.global_peak_count = Some(count);
                        group.pending_patch.global_peak_min_sep_cents = Some(min_sep);
                    }
                    _ => ctx.warn_live_builder(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_global_peaks_sep_int = ctx.clone();
        engine.register_fn(
            "global_peaks",
            move |handle: GroupHandle,
                  count: INT,
                  min_sep_cents: INT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_global_peaks_sep_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("global_peaks ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let min_sep = min_sep_cents as f32;
                match group.status {
                    GroupStatus::Draft => group.spec.set_global_peaks(count, min_sep),
                    GroupStatus::Live => {
                        group.spec.set_global_peaks(count, min_sep);
                        group.pending_patch.global_peak_count = Some(count);
                        group.pending_patch.global_peak_min_sep_cents = Some(min_sep);
                    }
                    _ => ctx.warn_live_builder(handle.id, "global_peaks"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_ratio_candidates = ctx.clone();
        engine.register_fn(
            "ratio_candidates",
            move |handle: GroupHandle, count: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_ratio_candidates
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("ratio_candidates ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_ratio_candidates(count),
                    GroupStatus::Live => {
                        group.spec.set_ratio_candidates(count);
                        group.pending_patch.ratio_candidate_count = Some(count);
                        group.pending_patch.use_ratio_candidates = Some(count > 0);
                    }
                    _ => ctx.warn_live_builder(handle.id, "ratio_candidates"),
                }
                Ok(handle)
            },
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "window_cents",
            SpeciesSpec::set_window_cents,
            patch_window_cents,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "top_k",
            SpeciesSpec::set_top_k,
            patch_top_k,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "temperature",
            SpeciesSpec::set_temperature,
            patch_temperature,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "sigma_cents",
            SpeciesSpec::set_sigma_cents,
            patch_sigma_cents,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "random_candidates",
            SpeciesSpec::set_random_candidates,
            patch_random_candidates,
            None,
        );
        let ctx_for_group_move_cost_time_scale = ctx.clone();
        engine.register_fn(
            "move_cost_time_scale",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_move_cost_time_scale
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "move_cost_time_scale ignored for unknown group {}",
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
                        ctx.warn_live_builder(handle.id, "move_cost_time_scale");
                        return Ok(handle);
                    }
                };
                match group.status {
                    GroupStatus::Draft => group.spec.control.set_move_cost_time_scale(value),
                    GroupStatus::Live => {
                        group.spec.control.set_move_cost_time_scale(value);
                        group.pending_patch.move_cost_time_scale = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "move_cost_time_scale"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_loo_harmonics = ctx.clone();
        engine.register_fn(
            "leave_self_out_harmonics",
            move |handle: GroupHandle, value: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_loo_harmonics
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "leave_self_out_harmonics ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_leave_self_out_harmonics(value),
                    GroupStatus::Live => {
                        group.spec.set_leave_self_out_harmonics(value);
                        group.pending_patch.leave_self_out_harmonics = Some(value);
                    }
                    _ => ctx.warn_live_builder(handle.id, "leave_self_out_harmonics"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_apply_mode = ctx.clone();
        engine.register_fn(
            "pitch_apply_mode",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_apply_mode
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_apply_mode ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let lowered = name.trim().to_ascii_lowercase();
                let mode = match lowered.as_str() {
                    "gate_snap" | "gatesnap" | "snap" => PitchApplyMode::GateSnap,
                    "glide" | "gliss" | "glissando" => PitchApplyMode::Glide,
                    _ => {
                        ctx.warn_live_builder(handle.id, "pitch_apply_mode");
                        return Ok(handle);
                    }
                };
                match group.status {
                    GroupStatus::Draft => group.spec.control.set_pitch_apply_mode(mode),
                    GroupStatus::Live => {
                        group.spec.control.set_pitch_apply_mode(mode);
                        group.pending_patch.pitch_apply_mode = Some(mode);
                    }
                    _ => ctx.warn_live_builder(handle.id, "pitch_apply_mode"),
                }
                Ok(handle)
            },
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "pitch_glide",
            SpeciesSpec::set_pitch_glide_tau_sec,
            patch_pitch_glide_tau,
            None,
        );
        let ctx_for_group_brain = ctx.clone();
        engine.register_fn(
            "brain",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_brain.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("brain ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_brain(name),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "brain"),
                    _ => ctx.warn_live_builder(handle.id, "brain"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_mode = ctx.clone();
        engine.register_fn(
            "pitch_mode",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_mode
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_mode ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_pitch_mode(name),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "pitch_mode"),
                    _ => ctx.warn_live_builder(handle.id, "pitch_mode"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_pitch_core = ctx.clone();
        engine.register_fn(
            "pitch_core",
            move |handle: GroupHandle, name: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_pitch_core
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("pitch_core ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_pitch_core(name),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "pitch_core"),
                    _ => ctx.warn_live_builder(handle.id, "pitch_core"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_sustain = ctx.clone();
        engine.register_fn(
            "sustain",
            move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_sustain.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("sustain ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_phonation(PhonationKind::Sustain);
                    }
                    _ => ctx.warn_live_builder(handle.id, "sustain"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_repeat = ctx.clone();
        engine.register_fn(
            "repeat",
            move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_repeat.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("repeat ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_phonation(PhonationKind::Repeat);
                    }
                    _ => ctx.warn_live_builder(handle.id, "repeat"),
                }
                Ok(handle)
            },
        );
        // Tier 2: explicit when/duration (group, draft-only)
        macro_rules! register_group_draft_fn {
            ($name:expr, $ctx:expr, $engine:expr, |$spec:ident| $body:expr) => {{
                let ctx_clone = $ctx.clone();
                $engine.register_fn(
                    $name,
                    move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                        let mut ctx = ctx_clone.lock().expect("lock script context");
                        let Some(group) = ctx.groups.get_mut(&handle.id) else {
                            warn!("{} ignored for unknown group {}", $name, handle.id);
                            return Ok(handle);
                        };
                        match group.status {
                            GroupStatus::Draft => {
                                let $spec = &mut group.spec;
                                $body;
                            }
                            _ => ctx.warn_live_builder(handle.id, $name),
                        }
                        Ok(handle)
                    },
                );
            }};
        }
        macro_rules! register_group_draft_fn1 {
            ($name:expr, $ctx:expr, $engine:expr, |$spec:ident, $a:ident: $at:ty| $body:expr) => {{
                let ctx_clone = $ctx.clone();
                $engine.register_fn(
                    $name,
                    move |handle: GroupHandle,
                          $a: $at|
                          -> Result<GroupHandle, Box<EvalAltResult>> {
                        let mut ctx = ctx_clone.lock().expect("lock script context");
                        let Some(group) = ctx.groups.get_mut(&handle.id) else {
                            warn!("{} ignored for unknown group {}", $name, handle.id);
                            return Ok(handle);
                        };
                        match group.status {
                            GroupStatus::Draft => {
                                let $spec = &mut group.spec;
                                $body;
                            }
                            _ => ctx.warn_live_builder(handle.id, $name),
                        }
                        Ok(handle)
                    },
                );
            }};
        }
        macro_rules! register_group_draft_fn2 {
            ($name:expr, $ctx:expr, $engine:expr, |$spec:ident, $a:ident: $at:ty, $b:ident: $bt:ty| $body:expr) => {{
                let ctx_clone = $ctx.clone();
                $engine.register_fn(
                    $name,
                    move |handle: GroupHandle,
                          $a: $at,
                          $b: $bt|
                          -> Result<GroupHandle, Box<EvalAltResult>> {
                        let mut ctx = ctx_clone.lock().expect("lock script context");
                        let Some(group) = ctx.groups.get_mut(&handle.id) else {
                            warn!("{} ignored for unknown group {}", $name, handle.id);
                            return Ok(handle);
                        };
                        match group.status {
                            GroupStatus::Draft => {
                                let $spec = &mut group.spec;
                                $body;
                            }
                            _ => ctx.warn_live_builder(handle.id, $name),
                        }
                        Ok(handle)
                    },
                );
            }};
        }
        register_group_draft_fn!("once", ctx, engine, |s| s.set_when_once());
        register_group_draft_fn!("mute", ctx, engine, |s| s
            .control
            .body
            .routing
            .to_listener = false);
        register_group_draft_fn!("unperceived", ctx, engine, |s| s
            .control
            .body
            .routing
            .to_voices = false);
        register_group_draft_fn1!("pulse", ctx, engine, |s, rate: FLOAT| s
            .set_when_pulse(rate as f32));
        register_group_draft_fn!("while_alive", ctx, engine, |s| s.set_duration_while_alive());
        register_group_draft_fn1!("gates", ctx, engine, |s, n: INT| s
            .set_duration_gates(n.max(1) as u32));
        register_group_draft_fn!("field", ctx, engine, |s| s.set_duration_field());
        register_group_draft_fn1!("sync", ctx, engine, |s, depth: FLOAT| s
            .set_sync(depth as f32));
        register_group_draft_fn1!("social", ctx, engine, |s, coupling: FLOAT| s
            .set_social(coupling as f32));
        register_group_draft_fn2!("field_window", ctx, engine, |s, min: FLOAT, max: FLOAT| s
            .set_field_window(min as f32, max as f32));
        register_group_draft_fn2!("field_curve", ctx, engine, |s, k: FLOAT, x0: FLOAT| s
            .set_field_curve(k as f32, x0 as f32));
        register_group_draft_fn1!("field_drop", ctx, engine, |s, gain: FLOAT| s
            .set_field_drop(gain as f32));
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "brightness",
            SpeciesSpec::set_brightness,
            patch_timbre_brightness,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "spread",
            SpeciesSpec::set_spread,
            patch_timbre_spread,
            None,
        );
        register_group_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "unison",
            SpeciesSpec::set_unison,
            patch_timbre_unison,
            None,
        );
        let ctx_for_group_modes = ctx.clone();
        engine.register_fn(
            "modes",
            move |handle: GroupHandle,
                  pattern: ModePattern|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_modes.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("modes ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_modes(pattern),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "modes"),
                    _ => ctx.warn_live_builder(handle.id, "modes"),
                }
                Ok(handle)
            },
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "metabolism",
            SpeciesSpec::set_metabolism,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "initial_energy",
            SpeciesSpec::set_initial_energy,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "recharge_rate",
            SpeciesSpec::set_recharge_rate,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "action_cost",
            SpeciesSpec::set_action_cost,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "continuous_recharge_rate",
            SpeciesSpec::set_continuous_recharge_rate,
        );
        register_group_draft_pair_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "survival_signal",
            SpeciesSpec::set_survival_signal,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "dissonance_cost",
            SpeciesSpec::set_dissonance_cost,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "energy_cap",
            SpeciesSpec::set_energy_cap,
        );
        let ctx_for_group_adsr = ctx.clone();
        engine.register_fn(
            "adsr",
            move |handle: GroupHandle,
                  a: FLOAT,
                  d: FLOAT,
                  s: FLOAT,
                  r: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_adsr.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("adsr ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_adsr(a as f32, d as f32, s as f32, r as f32);
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "adsr"),
                    _ => ctx.warn_live_builder(handle.id, "adsr"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_rhythm_coupling = ctx.clone();
        engine.register_fn(
            "rhythm_coupling",
            move |handle: GroupHandle, mode: &str| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_rhythm_coupling
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("rhythm_coupling ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_rhythm_coupling(mode),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "rhythm_coupling"),
                    _ => ctx.warn_live_builder(handle.id, "rhythm_coupling"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_rhythm_coupling_vitality = ctx.clone();
        engine.register_fn(
            "rhythm_coupling_vitality",
            move |handle: GroupHandle,
                  lambda_v: FLOAT,
                  v_floor: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_rhythm_coupling_vitality
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!(
                        "rhythm_coupling_vitality ignored for unknown group {}",
                        handle.id
                    );
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group
                        .spec
                        .set_rhythm_coupling_vitality(lambda_v as f32, v_floor as f32),
                    GroupStatus::Live => {
                        ctx.warn_live_builder(handle.id, "rhythm_coupling_vitality")
                    }
                    _ => ctx.warn_live_builder(handle.id, "rhythm_coupling_vitality"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_rhythm_reward = ctx.clone();
        engine.register_fn(
            "rhythm_reward",
            move |handle: GroupHandle,
                  rho_t: FLOAT,
                  metric: &str|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_rhythm_reward
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("rhythm_reward ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_rhythm_reward(rho_t as f32, metric),
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "rhythm_reward"),
                    _ => ctx.warn_live_builder(handle.id, "rhythm_reward"),
                }
                Ok(handle)
            },
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "rhythm_freq",
            SpeciesSpec::set_rhythm_freq,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "rhythm_sensitivity",
            SpeciesSpec::set_rhythm_sensitivity,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "k_omega",
            SpeciesSpec::set_k_omega,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "base_sigma",
            SpeciesSpec::set_base_sigma,
        );
        let ctx_for_group_gate_thresholds = ctx.clone();
        engine.register_fn(
            "gate_thresholds",
            move |handle: GroupHandle,
                  env_open: FLOAT,
                  mag: FLOAT,
                  alpha: FLOAT,
                  beta: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_gate_thresholds
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("gate_thresholds ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => group.spec.set_gate_thresholds(
                        env_open as f32,
                        mag as f32,
                        alpha as f32,
                        beta as f32,
                    ),
                    _ => ctx.warn_live_builder(handle.id, "gate_thresholds"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_respawn_random = ctx.clone();
        engine.register_fn(
            "respawn_random",
            move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_respawn_random
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("respawn_random ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.respawn_policy = RespawnPolicy::Random;
                        group.spec.respawn_policy = RespawnPolicy::Random;
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "respawn_random"),
                    _ => ctx.warn_live_builder(handle.id, "respawn_random"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_respawn_hereditary = ctx.clone();
        engine.register_fn(
            "respawn_hereditary",
            move |handle: GroupHandle,
                  sigma_oct: FLOAT|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_respawn_hereditary
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("respawn_hereditary ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let sigma_oct = sigma_oct as f32;
                let sigma_oct = if sigma_oct.is_finite() {
                    sigma_oct.max(0.0)
                } else {
                    0.0
                };
                match group.status {
                    GroupStatus::Draft => {
                        let policy = RespawnPolicy::Hereditary { sigma_oct };
                        group.respawn_policy = policy;
                        group.spec.respawn_policy = policy;
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "respawn_hereditary"),
                    _ => ctx.warn_live_builder(handle.id, "respawn_hereditary"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_respawn_hereditary_int = ctx.clone();
        engine.register_fn(
            "respawn_hereditary",
            move |handle: GroupHandle, sigma_oct: INT| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_respawn_hereditary_int
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("respawn_hereditary ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                let sigma_oct = (sigma_oct as f32).max(0.0);
                match group.status {
                    GroupStatus::Draft => {
                        let policy = RespawnPolicy::Hereditary { sigma_oct };
                        group.respawn_policy = policy;
                        group.spec.respawn_policy = policy;
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "respawn_hereditary"),
                    _ => ctx.warn_live_builder(handle.id, "respawn_hereditary"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_respawn_peak_bias = ctx.clone();
        engine.register_fn(
            "respawn_peak_bias",
            move |handle: GroupHandle| -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_respawn_peak_bias
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("respawn_peak_bias ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        let policy = RespawnPolicy::PeakBiased {
                            config: RespawnPeakBiasConfig::default(),
                        };
                        group.respawn_policy = policy;
                        group.spec.respawn_policy = policy;
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "respawn_peak_bias"),
                    _ => ctx.warn_live_builder(handle.id, "respawn_peak_bias"),
                }
                Ok(handle)
            },
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "respawn_capacity",
            SpeciesSpec::set_respawn_capacity,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "respawn_min_c_level",
            SpeciesSpec::set_respawn_min_c_level,
        );
        register_group_draft_numeric_overloads(
            &mut engine,
            ctx.clone(),
            "respawn_background_death_rate",
            SpeciesSpec::set_respawn_background_death_rate,
        );
        let ctx_for_group_respawn_settle = ctx.clone();
        engine.register_fn(
            "respawn_settle",
            move |handle: GroupHandle,
                  strategy: SpawnStrategy|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_respawn_settle
                    .lock()
                    .expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("respawn_settle ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.spec.set_respawn_settle_strategy(strategy);
                    }
                    _ => ctx.warn_live_builder(handle.id, "respawn_settle"),
                }
                Ok(handle)
            },
        );
        let ctx_for_group_place = ctx.clone();
        engine.register_fn(
            "place",
            move |handle: GroupHandle,
                  strategy: SpawnStrategy|
                  -> Result<GroupHandle, Box<EvalAltResult>> {
                let mut ctx = ctx_for_group_place.lock().expect("lock script context");
                let Some(group) = ctx.groups.get_mut(&handle.id) else {
                    warn!("place ignored for unknown group {}", handle.id);
                    return Ok(handle);
                };
                match group.status {
                    GroupStatus::Draft => {
                        group.strategy = Some(strategy);
                    }
                    GroupStatus::Live => ctx.warn_live_builder(handle.id, "place"),
                    _ => ctx.warn_live_builder(handle.id, "place"),
                }
                Ok(handle)
            },
        );

        let ctx_for_set_harmonicity_mirror_weight = ctx.clone();
        engine.register_fn(
            "set_harmonicity_mirror_weight",
            move |_call_ctx: NativeCallContext, mirror: FLOAT| {
                let mut ctx = ctx_for_set_harmonicity_mirror_weight
                    .lock()
                    .expect("lock script context");
                let update = crate::core::landscape::LandscapeUpdate {
                    mirror: Some(mirror as f32),
                    ..crate::core::landscape::LandscapeUpdate::default()
                };
                let cursor = ctx.cursor;
                ctx.push_event(cursor, vec![Action::SetHarmonicityParams { update }]);
            },
        );
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

    pub fn load_script(path: &str) -> Result<Scenario, ScriptError> {
        let src = fs::read_to_string(path)
            .map_err(|err| ScriptError::new(format!("read script {path}: {err}"), None))?;
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
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
