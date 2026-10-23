use super::*;

#[test]
fn output_gain_is_scene_global_and_last_declaration_wins() {
    assert_eq!(run_script("").0.output_gain_db, 0.0);
    assert_eq!(run_script("output_gain(6);").0.output_gain_db, 6.0);
    let scenario = run_script("output_gain(3.5); wait(1); play(|| { output_gain(-2.5); });").0;
    assert_eq!(scenario.output_gain_db, -2.5);
    assert!(scenario.output_gain_position.is_some());
    assert!(scenario.events.iter().all(|event| {
        event
            .actions
            .iter()
            .all(|action| matches!(action, Action::Finish))
    }));
}

#[test]
fn output_gain_errors_preserve_declaration_positions() {
    for db in [
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        1e300,
        1000.0,
        -1000.0,
    ] {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let mut scope = rhai::Scope::new();
        scope.push("db", db);
        let error = engine
            .eval_with_scope::<Dynamic>(&mut scope, "\noutput_gain(db);")
            .expect_err("invalid gain");
        assert_eq!(error.position().line(), Some(2), "{error}");
        assert!(
            error.to_string().contains("finite, positive f32"),
            "{error}"
        );
        assert_eq!(ctx.lock().unwrap().scenario.output_gain_db, 0.0);
    }
}

#[test]
fn temporal_mode_is_a_single_pre_performance_setting() {
    use crate::scenario::TemporalMode;
    assert_eq!(run_script("").0.temporal_mode, TemporalMode::Off);
    assert_eq!(
        run_script("temporal_mode(\"off\");").0.temporal_mode,
        TemporalMode::Off
    );
    assert_eq!(
        run_script("temporal_mode(\"observe\"); create(sine(), 1);")
            .0
            .temporal_mode,
        TemporalMode::Observe
    );
    for script in [
        "temporal_mode(\"observe\"); temporal_mode(\"observe\");",
        "wait(1.0); temporal_mode(\"observe\");",
        "create(sine(), 1); temporal_mode(\"observe\");",
        "section(\"a\", || { temporal_mode(\"observe\"); });",
        "parallel([|| { temporal_mode(\"observe\"); }]);",
        "fn mode() { temporal_mode(\"observe\"); } mode();",
    ] {
        assert!(
            run_script_err(script)
                .to_string()
                .contains("temporal_mode must be declared"),
            "{script}"
        );
    }
    assert!(
        run_script_err("temporal_mode(\"participate\");")
            .to_string()
            .contains("temporal_mode expects off or observe")
    );
    assert!(
        run_script_err("temporal_mode(\"unknown\");")
            .to_string()
            .contains("expects off or observe")
    );

    let ctx = Arc::new(Mutex::new(ScriptContext::default()));
    let engine = ScriptHost::create_engine(ctx.clone());
    assert!(
        engine
            .eval::<Dynamic>("temporal_mode(\"participate\");")
            .is_err()
    );
    assert_eq!(
        ctx.lock().unwrap().scenario.temporal_mode,
        TemporalMode::Off
    );
    let _ = engine
        .eval::<Dynamic>("temporal_mode(\"observe\");")
        .unwrap();
}

#[test]
fn intrinsic_rate_resolves_at_creation_independent_of_builder_order() {
    for body in ["sine", "harmonic", "modal"] {
        for preset in ["entrained()", "steady()", "flow()"] {
            for chain in [
                format!("intrinsic_rate(1.0/2.128).sustain().{preset}"),
                format!("sustain().intrinsic_rate(1.0/2.128).{preset}"),
                format!("sustain().{preset}.intrinsic_rate(1.0/2.128)"),
            ] {
                for script in [
                    format!("create({body}().{chain},1);"),
                    format!("place({body}().{chain},at(330.0));"),
                ] {
                    let spec = first_spawn_spec_for_script(&script);
                    let PhonationTiming::Coupled(timing) = spec.control.phonation.spec.timing
                    else {
                        panic!("{script}")
                    };
                    assert_eq!(timing.relation, RhythmRelation::Participating);
                    assert_eq!(timing.base_rate_hz, (1.0 / 2.128) as f32);
                    assert!(spec.control.phonation.spec.is_sustained_participation());
                }
            }
        }
    }
}

#[test]
fn cli_seed_is_used_when_script_has_no_seed_call() {
    let (scenario, _warnings) = run_script_with_seed(
        r#"
            create(sine(), 1);
            flush();
        "#,
        123_456_789,
    );
    assert_eq!(scenario.seed, 123_456_789);
}

#[test]
fn script_seed_call_overrides_cli_seed() {
    let (scenario, _warnings) = run_script_with_seed(
        r#"
            seed(42);
            create(sine(), 1);
            flush();
        "#,
        123_456_789,
    );
    assert_eq!(scenario.seed, 42);
}

#[test]
fn population_is_spawned_without_wait_or_flush() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 1);
        "#,
    );
    let has_spawn = scenario
        .events
        .iter()
        .any(|ev| ev.actions.iter().any(|a| matches!(a, Action::Spawn { .. })));
    assert!(has_spawn, "place/create is the population boundary");
}

#[test]
fn flush_does_not_delay_or_advance_immediate_spawn() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 1);
            flush();
            wait(1.0);
        "#,
    );
    let mut spawn_time = None;
    let mut finish_time = None;
    for (time, action) in action_times(&scenario) {
        match action {
            Action::Spawn { .. } => spawn_time = Some(time),
            Action::Finish => finish_time = Some(time),
            _ => {}
        }
    }
    assert_eq!(spawn_time, Some(0.0));
    assert_eq!(finish_time, Some(1.0));
}

#[test]
fn scene_scope_releases_live_populations() {
    let (scenario, _warnings) = run_script(
        r#"
            section("alpha", || {
                let population = create(sine(), 1);
                flush();
                wait(0.5);
            });
        "#,
    );
    let mut release_time = None;
    for (time, action) in action_times(&scenario) {
        if matches!(action, Action::ReleasePopulation { .. }) {
            release_time = Some(time);
        }
    }
    assert_eq!(release_time, Some(0.5));
}

#[test]
fn scope_releases_population_even_without_wait() {
    let (scenario, _warnings) = run_script(
        r#"
            section("alpha", || { create(sine(), 1); });
        "#,
    );
    assert!(scenario.events.iter().any(|event| {
        event
            .actions
            .iter()
            .any(|action| matches!(action, Action::ReleasePopulation { .. }))
    }));
}

#[test]
fn scope_emits_pending_patch_before_automatic_release() {
    let (scenario, _warnings) = run_script(
        r#"
            section("alpha", || {
                let population = place(sine().amp(0.1), at(220.0));
                wait(0.5);
                population.amp(0.3);
            });
        "#,
    );
    let actions: Vec<&Action> = scenario
        .events
        .iter()
        .filter(|event| (event.time - 0.5).abs() <= f32::EPSILON)
        .flat_map(|event| event.actions.iter())
        .collect();
    let update_idx = actions
        .iter()
        .position(|action| {
            matches!(
                action,
                Action::UpdatePopulation { patch, .. } if patch.amp == Some(0.3)
            )
        })
        .expect("scope should emit its pending patch");
    let release_idx = actions
        .iter()
        .position(|action| matches!(action, Action::ReleasePopulation { .. }))
        .expect("scope should release its population");
    assert!(update_idx < release_idx);
}

#[test]
fn release_is_terminal_for_live_patches() {
    let (scenario, _warnings) = run_script(
        r#"
            let population = place(sine(), at(220.0));
            release(population);
            population.amp(0.5);
            flush();
        "#,
    );
    assert!(!scenario.events.iter().any(|event| {
        event
            .actions
            .iter()
            .any(|action| matches!(action, Action::UpdatePopulation { .. }))
    }));
    assert!(scenario.events.iter().any(|event| {
        event
            .actions
            .iter()
            .any(|action| matches!(action, Action::ReleasePopulation { .. }))
    }));
}

#[test]
fn spawn_order_is_population_id_order() {
    let (scenario, _warnings) = run_script(
        r#"
            let a = create(sine(), 1);
            let b = create(sine(), 1);
            flush();
        "#,
    );
    let mut spawns = Vec::new();
    for event in &scenario.events {
        for action in &event.actions {
            if let Action::Spawn {
                population_id, ids, ..
            } = action
            {
                spawns.push((event.time, *population_id, ids.clone()));
            }
        }
    }
    assert_eq!(spawns.len(), 2);
    assert_eq!(spawns[0].0, 0.0);
    assert_eq!(spawns[0].1, 1);
    assert_eq!(spawns[0].2, vec![1]);
    assert_eq!(spawns[1].0, 0.0);
    assert_eq!(spawns[1].1, 2);
    assert_eq!(spawns[1].2, vec![2]);
}

#[test]
fn place_spawns_population_and_same_time_patch_is_separate() {
    let (scenario, _warnings) = run_script(
        r#"
            let p = place(
                harmonic().amp(0.04).send(habitat_bus | presentation_bus),
                consonance(90.0, 900.0).count(3).spacing(0.8)
            );
            p.amp(0.02);
            flush();
        "#,
    );
    let mut spawn = None;
    let mut patch = None;
    for event in &scenario.events {
        for action in &event.actions {
            match action {
                Action::Spawn {
                    ids,
                    spec,
                    strategy,
                    ..
                } => {
                    spawn = Some((
                        event.time,
                        ids.len(),
                        spec.control.clone(),
                        strategy.clone(),
                    ));
                }
                Action::UpdatePopulation { patch: update, .. } => {
                    patch = update.amp;
                }
                _ => {}
            }
        }
    }
    let (time, count, control, strategy) = spawn.expect("spawn action");
    assert_eq!(time, 0.0);
    assert_eq!(count, 3);
    assert_eq!(control.body.amp, 0.04);
    assert_eq!(patch, Some(0.02));
    assert!(control.body.routing.to_habitat);
    assert!(control.body.routing.to_presentation);
    assert!(matches!(
        strategy,
        Some(SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            ..
        })
    ));
}

#[test]
fn flush_events_have_increasing_order_at_same_time() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 1);
            flush();
            create(sine(), 1);
            flush();
        "#,
    );
    let mut orders = Vec::new();
    for event in &scenario.events {
        if (event.time - 0.0).abs() <= f32::EPSILON {
            orders.push(event.order);
        }
    }
    assert!(orders.len() >= 2);
    for pair in orders.windows(2) {
        assert!(pair[0] < pair[1]);
    }
}

#[test]
fn explicit_pitch_apply_mode_is_order_independent() {
    for script in [
        r#"create(sine().pitch_apply_mode("glide").metric().seek_consonance(), 1); flush();"#,
        r#"create(sine().metric().seek_consonance().pitch_apply_mode("glide"), 1); flush();"#,
    ] {
        let spec = first_spawn_spec_for_script(script);
        assert_eq!(
            spec.control.pitch.pitch_apply_mode,
            PitchApplyMode::Glide,
            "explicit pitch_apply_mode must win regardless of call order"
        );
    }
}

#[test]
fn population_live_update_last_write_wins_within_flush() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.amp(0.2).amp(0.8);
            flush();
            let g = g.avoid_neighbors(1.0, 35.0).avoid_neighbors(1.0);
            flush();
            let g = g.avoid_neighbors(1.0).avoid_neighbors(1.0, 35.0);
            flush();
        "#,
    );

    let updates = scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
        .filter_map(|action| match action {
            Action::UpdatePopulation { patch, .. } => Some(patch.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();

    assert_eq!(updates.len(), 3, "expected three live update flushes");
    assert_eq!(updates[0].amp, Some(0.8));
    assert_eq!(updates[1].crowding_strength, Some(1.0));
    assert_eq!(updates[1].crowding_sigma_cents, Some(35.0));
    assert_eq!(updates[2].crowding_strength, Some(1.0));
    assert_eq!(updates[2].crowding_sigma_cents, Some(35.0));
}

#[test]
fn flush_emits_update_before_release_for_same_population() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.amp(0.25);
            release(g);
            flush();
        "#,
    );

    let event = scenario
        .events
        .iter()
        .find(|event| {
            event
                .actions
                .iter()
                .any(|action| matches!(action, Action::UpdatePopulation { .. }))
                && event
                    .actions
                    .iter()
                    .any(|action| matches!(action, Action::ReleasePopulation { .. }))
        })
        .expect("event with both update and release");

    let update_idx = event
        .actions
        .iter()
        .position(|action| matches!(action, Action::UpdatePopulation { .. }))
        .expect("update action");
    let release_idx = event
        .actions
        .iter()
        .position(|action| matches!(action, Action::ReleasePopulation { .. }))
        .expect("release action");
    assert!(
        update_idx < release_idx,
        "flush must emit update before release within same event"
    );
}

#[test]
fn set_control_update_mode_updates_scenario() {
    let (scenario, _warnings) = run_script(
        r#"
            control_update_mode("sequential_rotating");
        "#,
    );
    assert_eq!(
        scenario.control_update_mode,
        ControlUpdateMode::SequentialRotating
    );
}

#[test]
fn set_scaffold_shared_updates_scenario() {
    let (scenario, _warnings) = run_script(
        r#"
            scaffold_shared(2.5);
        "#,
    );
    assert!(matches!(
        scenario.scaffold,
        ScaffoldConfig::Shared { freq_hz } if (freq_hz - 2.5).abs() <= 1e-6
    ));
}

#[test]
fn set_scaffold_scrambled_updates_scenario() {
    let (scenario, _warnings) = run_script(
        r#"
            scaffold_scrambled(3.0, 17);
        "#,
    );
    assert!(matches!(
        scenario.scaffold,
        ScaffoldConfig::Scrambled { freq_hz, seed }
            if (freq_hz - 3.0).abs() <= 1e-6 && seed == 17
    ));
}

#[test]
fn viability_scope_total_overrides_environment_default() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                harmonic()                    .endurance(10.0)
                    .consonance_viability(0.3, 0.8)
                    .viability_scope("total"),
                1
            );
            flush();
        "#,
    );
    let spawn = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.clone()),
            _ => None,
        })
        .expect("spawn action");
    let ArticulationCoreConfig::Entrain { lifecycle, .. } = spawn.articulation else {
        panic!("expected entrain articulation");
    };
    let LifecycleConfig::Sustain {
        selection_environment,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert!(!selection_environment);
}

#[test]
fn population_spec_viability_scope_total_overrides_environment_default() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                harmonic()
                    .endurance(10.0)
                    .consonance_viability(0.3, 0.8)
                    .viability_scope("total"),
                1
            );
        "#,
    );
    let spawn = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.clone()),
            _ => None,
        })
        .expect("spawn action");
    let ArticulationCoreConfig::Entrain { lifecycle, .. } = spawn.articulation else {
        panic!("expected entrain articulation");
    };
    let LifecycleConfig::Sustain {
        selection_environment,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert!(!selection_environment);
}

#[test]
fn live_population_brightness_emits_timbre_patch() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(harmonic(), 1);
            flush();
            g.brightness(0.25);
            flush();
        "#,
    );
    let patch = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::UpdatePopulation { patch, .. } => Some(patch.clone()),
            _ => None,
        })
        .expect("update action");
    assert_eq!(patch.timbre_brightness, Some(0.25));
}

#[test]
fn live_population_spread_and_unison_emit_timbre_patch() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(harmonic(), 1);
            flush();
            g.spread(0.3).unison(4);
            flush();
        "#,
    );
    let patch = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::UpdatePopulation { patch, .. } => Some(patch.clone()),
            _ => None,
        })
        .expect("update action");
    assert_eq!(patch.timbre_spread, Some(0.3));
    assert_eq!(patch.timbre_unison, Some(4));
}

#[test]
fn measure_accent_tuning_is_bounded_and_order_independent() {
    for preset in ["metric", "entrained", "flow"] {
        for chain in [
            format!("{preset}().measure_accent(0.6)"),
            format!("measure_accent(0.6).{preset}()"),
        ] {
            let spawn = first_spawn_spec_for_script(&format!(
                "create(sine().{chain}.cycles(1), 1); flush();"
            ));
            let PhonationTiming::Coupled(spec) = spawn.control.phonation.spec.timing else {
                panic!("expected coupled timing");
            };
            assert!((spec.measure_accent - 0.6).abs() < 1e-6);
        }
    }
    for (amount, expected) in [("-1.0", 0.0), ("2.0", 1.0)] {
        let spawn = first_spawn_spec_for_script(&format!(
            "create(sine().measure_accent({amount}), 1); flush();"
        ));
        let PhonationTiming::Coupled(spec) = spawn.control.phonation.spec.timing else {
            panic!("expected coupled timing");
        };
        assert_eq!(spec.measure_accent, expected);
    }
}

#[test]
fn metric_and_pulse_tuning_are_order_independent() {
    for (label, script) in [
        (
            "metric then tuning",
            r#"
                create(sine().metric().entrainment(0.8).rhythm_role("accent").cycles(2), 1);
                flush();
            "#,
        ),
        (
            "tuning then metric",
            r#"
                create(sine().entrainment(0.8).rhythm_role("accent").metric().cycles(2), 1);
                flush();
            "#,
        ),
    ] {
        let spawn = first_spawn_spec_for_script(script);
        let PhonationTiming::Coupled(spec) = spawn.control.phonation.spec.timing else {
            panic!("{label}: expected coupled timing intent");
        };
        assert!((spec.coupling - 0.8).abs() <= 1e-6, "{label}");
        assert_eq!(spec.role, RhythmRole::Accent, "{label}");
    }

    for (label, script) in [
        (
            "pulse then tuning",
            r#"
                create(sine().repeat().pulse(2.0).pulse_lock(0.35).cycles(2), 1);
                flush();
            "#,
        ),
        (
            "tuning then pulse",
            r#"
                create(sine().pulse_lock(0.35).pulse(2.0).cycles(2), 1);
                flush();
            "#,
        ),
    ] {
        let spawn = first_spawn_spec_for_script(script);
        let PhonationTiming::Pulse { rate_hz, sync, .. } = &spawn.control.phonation.spec.timing
        else {
            panic!("{label}: expected pulse timing");
        };
        assert!((*rate_hz - 2.0).abs() <= 1e-6, "{label}");
        assert!((*sync - 0.35).abs() <= 1e-6, "{label}");
        assert!(
            matches!(
                spawn.control.phonation.spec.duration,
                DurationSpec::Gates(2)
            ),
            "{label}"
        );
    }
}

#[test]
fn adaptive_duration_tuning_is_order_independent() {
    for (label, script) in [
        (
            "duration then tuning",
            r#"
                create(
                    sine()
                        .once()
                        .adaptive_duration()
                        .duration_range(0.2, 0.8)
                        .duration_curve(3.0, 0.4)
                        .shorten_on_drop(0.6),
                    1
                );
                flush();
            "#,
        ),
        (
            "tuning then duration",
            r#"
                create(
                    sine()
                        .duration_range(0.2, 0.8)
                        .duration_curve(3.0, 0.4)
                        .shorten_on_drop(0.6)
                        .adaptive_duration(),
                    1
                );
                flush();
            "#,
        ),
    ] {
        let spawn = first_spawn_spec_for_script(script);
        let DurationSpec::Field(field) = spawn.control.phonation.spec.duration else {
            panic!("{label}: expected adaptive field duration");
        };
        assert!((field.hold_min_theta - 0.2).abs() <= 1e-6, "{label}");
        assert!((field.hold_max_theta - 0.8).abs() <= 1e-6, "{label}");
        assert!((field.curve_k - 3.0).abs() <= 1e-6, "{label}");
        assert!((field.curve_x0 - 0.4).abs() <= 1e-6, "{label}");
        assert!((field.drop_gain - 0.6).abs() <= 1e-6, "{label}");
    }
}

#[test]
fn rearticulate_live_error_preserves_spec_pending_patch_and_future_template() {
    let ctx = Arc::new(Mutex::new(ScriptContext::with_seed(21)));
    let engine = ScriptHost::create_engine(ctx.clone());
    let mut scope = rhai::Scope::new();
    let _ = engine.eval_with_scope::<Dynamic>(&mut scope, "let p = place(sine().repeat().pulse(2.0).seek_consonance(), line(330.0, 330.0)); p.amp(0.2);").unwrap();
    let (mode, patch, event_count) = {
        let state = ctx.lock().unwrap();
        let population = state.populations.values().next().unwrap();
        (
            population.spec.spawn_spec().control.pitch.pitch_apply_mode,
            population.pending_patch.clone(),
            state.scenario.events.len(),
        )
    };
    let error = engine
        .eval_with_scope::<Dynamic>(&mut scope, "p.pitch_apply_mode(\"rearticulate\");")
        .expect_err("incompatible live designation");
    assert!(error.to_string().contains("Gated"));
    let state = ctx.lock().unwrap();
    let population = state.populations.values().next().unwrap();
    assert_eq!(
        population.spec.spawn_spec().control.pitch.pitch_apply_mode,
        mode
    );
    assert_eq!(population.pending_patch, patch);
    assert_eq!(state.scenario.events.len(), event_count);
}

#[test]
fn rearticulate_native_invalid_patch_keeps_voice_and_respawn_template() {
    let (scenario, _) = run_script_with_seed(
        "create(sine().repeat().pulse(2.0).seek_consonance().respawn_random().respawn_background_death_rate(10000.0), 1);",
        21,
    );
    let mut community = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    community.set_seed(scenario.seed);
    let landscape = Landscape::default();
    for action in scenario.events.iter().flat_map(|e| &e.actions) {
        if matches!(
            action,
            Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
        ) {
            community.apply_action(action.clone(), &landscape);
        }
    }
    let patch = ControlUpdate {
        pitch_apply_mode: Some(PitchApplyMode::Rearticulate),
        ..Default::default()
    };
    let v = &mut community.voices[0];
    assert!(v.apply_patch(&patch).is_err());
    assert_eq!(
        v.base_control.pitch.pitch_apply_mode,
        PitchApplyMode::GateSnap
    );
    assert_eq!(
        v.effective_control.pitch.pitch_apply_mode,
        PitchApplyMode::GateSnap
    );
    let population_id = v.metadata.population_id;
    let founder = v.id();
    community.apply_action(
        Action::UpdatePopulation {
            population_id,
            patch,
        },
        &landscape,
    );
    assert_eq!(
        community.voices[0].effective_control.pitch.pitch_apply_mode,
        PitchApplyMode::GateSnap
    );
    community.advance(64, 0, 64.0 / 48_000.0, &landscape);
    community.cleanup_dead(0, 64.0 / 48_000.0, false, &landscape);
    assert!(community.voices.iter().any(|v| v.id() != founder));
    for child in community.voices.iter().filter(|v| v.id() != founder) {
        assert_eq!(
            child.effective_control.pitch.pitch_apply_mode,
            PitchApplyMode::GateSnap
        );
    }
}

#[test]
fn sustained_participation_explicit_builders_are_order_independent() {
    use crate::life::phonation_engine::PhonationEngine;
    for preset in ["entrained()", "steady()", "flow()"] {
        for pitch in ["", ".seek_consonance()"] {
            let a = first_spawn_spec_for_script(&format!(
                "create(harmonic().sustain().{preset}{pitch}, 1);"
            ));
            let b = first_spawn_spec_for_script(&format!(
                "create(harmonic().{preset}.sustain(){pitch}, 1);"
            ));
            assert_eq!(a.control.phonation.spec, b.control.phonation.spec);
            assert_eq!(
                format!("{:?}", a.articulation),
                format!("{:?}", b.articulation)
            );
            assert!(a.control.phonation.spec.is_sustained_participation());
            assert_eq!(a.control.pitch.pitch_apply_mode, PitchApplyMode::Glide);
            assert_eq!(b.control.pitch.pitch_apply_mode, PitchApplyMode::Glide);
            let config = PhonationEngine::from_spec(&a.control.phonation.spec, 21);
            assert_eq!(config.mode, crate::scenario::PhonationMode::Hold);
            assert!(config.uses_participation());
        }
    }
    for (chain, hold) in [
        ("", true),
        (".sustain()", true),
        (".entrained()", false),
        (".flow()", false),
    ] {
        let spec = first_spawn_spec_for_script(&format!("create(sine(){chain}, 1);"));
        assert!(!spec.control.phonation.spec.is_sustained_participation());
        let config = PhonationEngine::from_spec(&spec.control.phonation.spec, 21);
        assert_eq!(config.mode == crate::scenario::PhonationMode::Hold, hold);
        assert_eq!(
            spec.control.pitch.pitch_apply_mode,
            PitchApplyMode::GateSnap
        );
    }
    for chain in [
        "while_alive().entrained()",
        "while_alive().flow()",
        "entrained().while_alive()",
        "flow().while_alive()",
    ] {
        let spec = first_spawn_spec_for_script(&format!("create(sine().{chain}, 1);"));
        assert!(!spec.control.phonation.spec.is_sustained_participation());
    }
    for preset in ["pulse(2.0)", "repeat()"] {
        for (chain, hold) in [
            (format!("sustain().{preset}"), false),
            (format!("{preset}.sustain()"), true),
        ] {
            let spec = first_spawn_spec_for_script(&format!("create(sine().{chain}, 1);"));
            assert!(!spec.control.phonation.spec.is_sustained_participation());
            let config = PhonationEngine::from_spec(&spec.control.phonation.spec, 21);
            assert_eq!(config.mode == crate::scenario::PhonationMode::Hold, hold);
        }
    }
}

#[test]
fn synchronized_explicit_builders_and_finite_rest_are_order_independent() {
    use crate::life::phonation_engine::PhonationEngine;
    for chain in ["sustain().metric()", "metric().sustain()"] {
        for pitch in [
            "",
            ".pitch_apply_mode(\"gate_snap\").tie(1)",
            ".pitch_apply_mode(\"rearticulate\").rearticulate_above(50).articulation_persistence(0.5)",
        ] {
            let spec = first_spawn_spec_for_script(&format!("create(sine().{chain}{pitch},1);"));
            assert!(spec.control.phonation.spec.is_sustained_participation());
            let engine = PhonationEngine::from_spec(&spec.control.phonation.spec, 21);
            assert_eq!(engine.mode, crate::scenario::PhonationMode::Hold);
            assert!(!engine.uses_participation());
        }
    }
    let words = ["sustain()", "metric()", "cycles(1)", "rest()"];
    for a in 0..4 {
        for b in 0..4 {
            for c in 0..4 {
                for d in 0..4 {
                    if [a, b, c, d]
                        .iter()
                        .enumerate()
                        .any(|(i, v)| [a, b, c, d][..i].contains(v))
                    {
                        continue;
                    }
                    let chain = [words[a], words[b], words[c], words[d]].join(".");
                    let spec = first_spawn_spec_for_script(&format!(
                        "create(sine().{chain}.pitch_apply_mode(\"rearticulate\"),1);"
                    ));
                    assert!(
                        spec.control.phonation.spec.is_sustained_participation(),
                        "{chain}"
                    );
                    assert_eq!(spec.control.phonation.rest_periods, Some(1.));
                    assert_eq!(
                        spec.control.phonation.finite_duration,
                        Some(crate::scenario::DurationSpec::Gates(1))
                    );
                }
            }
        }
    }
}

#[test]
fn sustained_participation_rearticulate_order_live_and_respawn_are_valid() {
    for preset in ["entrained()", "steady()", "flow()"] {
        for chain in [
            format!("sustain().{preset}.pitch_apply_mode(\"rearticulate\").rearticulate_above(50)"),
            format!("rearticulate_above(50).pitch_apply_mode(\"rearticulate\").{preset}.sustain()"),
            format!("{preset}.pitch_apply_mode(\"rearticulate\").sustain().rearticulate_above(50)"),
        ] {
            let spec = first_spawn_spec_for_script(&format!("create(sine().{chain}, 1);"));
            assert!(spec.control.phonation.spec.is_sustained_participation());
            assert_eq!(
                spec.control.pitch.pitch_apply_mode,
                PitchApplyMode::Rearticulate
            );
            assert_eq!(spec.control.pitch.rearticulate_above_cents, 50.);
        }
        let (scenario, _) = run_script_with_seed(
            &format!(
                "let p = create(sine().sustain().{preset}.seek_consonance().respawn_random().respawn_background_death_rate(10000.0),1); p.pitch_apply_mode(\"rearticulate\"); p.rearticulate_above(100); flush();"
            ),
            21,
        );
        let mut community = Community::new(Timebase {
            fs: 48000.,
            hop: 257,
        });
        community.set_seed(scenario.seed);
        let landscape = Landscape::default();
        for action in scenario.events.iter().flat_map(|e| &e.actions) {
            if matches!(
                action,
                Action::Spawn { .. }
                    | Action::SetRespawnPolicy { .. }
                    | Action::UpdatePopulation { .. }
            ) {
                community.apply_action(action.clone(), &landscape);
            }
        }
        let founder = community.voices[0].id();
        community.advance(257, 0, 257. / 48000., &landscape);
        community.cleanup_dead(0, 257. / 48000., false, &landscape);
        let children: Vec<_> = community
            .voices
            .iter()
            .filter(|v| v.id() != founder)
            .collect();
        assert!(!children.is_empty());
        for child in children {
            assert_eq!(
                child.effective_control.pitch.pitch_apply_mode,
                PitchApplyMode::Rearticulate
            );
            assert_eq!(child.effective_control.pitch.rearticulate_above_cents, 100.);
            assert!(
                child
                    .effective_control
                    .phonation
                    .spec
                    .is_sustained_participation()
            );
        }
    }
}

#[test]
fn boundary_script_order_sanitize_and_outside_mode_ignore() {
    for chain in [
        "rearticulate_above(50).pitch_apply_mode(\"rearticulate\")",
        "pitch_apply_mode(\"rearticulate\").rearticulate_above(50)",
    ] {
        let (scenario, _) = run_script(&format!("create(sine().sustain().{chain}, 1);"));
        let spec = scenario
            .events
            .iter()
            .flat_map(|e| &e.actions)
            .find_map(|a| match a {
                Action::Spawn { spec, .. } => Some(spec),
                _ => None,
            })
            .unwrap();
        assert_eq!(spec.control.pitch.rearticulate_above_cents, 50.0);
    }
    let ctx = Arc::new(Mutex::new(ScriptContext::with_seed(21)));
    let engine = ScriptHost::create_engine(ctx.clone());
    let mut scope = rhai::Scope::new();
    let _ = engine.eval_with_scope::<Dynamic>(&mut scope, "let p = create(sine().sustain().pitch_apply_mode(\"glide\").rearticulate_above(50), 1); p.rearticulate_above(100); p.amp(0.2);").unwrap();
    {
        let state = ctx.lock().unwrap();
        let p = state.populations.values().next().unwrap();
        assert_eq!(p.spec.control.pitch.rearticulate_above_cents, 0.0);
        assert_eq!(p.pending_patch.rearticulate_above_cents, None);
    }
    let _ = engine
        .eval_with_scope::<Dynamic>(
            &mut scope,
            "p.pitch_apply_mode(\"rearticulate\"); p.rearticulate_above(100);",
        )
        .unwrap();
    {
        let state = ctx.lock().unwrap();
        let p = state.populations.values().next().unwrap();
        assert_eq!(p.spec.control.pitch.rearticulate_above_cents, 100.0);
        assert_eq!(p.pending_patch.rearticulate_above_cents, Some(100.0));
    }
    let mut pitch = crate::scenario::control::PitchControl::default();
    for value in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        pitch.set_rearticulate_above_cents_clamped(value);
        assert_eq!(pitch.rearticulate_above_cents, 0.0);
    }
    pitch.set_rearticulate_above_cents_clamped(f32::MAX);
    assert_eq!(pitch.rearticulate_above_cents, f32::MAX);
}

#[test]
fn deferred_participation_script_order_and_live_gate_snap_are_valid() {
    for preset in ["entrained()", "steady()", "flow()"] {
        for chain in [
            format!("sustain().{preset}.pitch_apply_mode(\"gate_snap\")"),
            format!("pitch_apply_mode(\"gate_snap\").{preset}.sustain()"),
            format!("{preset}.pitch_apply_mode(\"gate_snap\").sustain()"),
        ] {
            let (scenario, _) =
                run_script(&format!("create(sine().{chain}.seek_consonance(), 1);"));
            let spec = scenario
                .events
                .iter()
                .flat_map(|e| &e.actions)
                .find_map(|a| match a {
                    Action::Spawn { spec, .. } => Some(spec),
                    _ => None,
                })
                .unwrap();
            assert_eq!(
                spec.control.pitch.pitch_apply_mode,
                PitchApplyMode::GateSnap
            );
            assert!(spec.control.phonation.spec.is_sustained_participation());
        }
        run_script(&format!(
            "let p = create(sine().sustain().{preset}.seek_consonance(), 1); p.pitch_apply_mode(\"gate_snap\"); wait(0.1); p.pitch_apply_mode(\"glide\"); flush();"
        ));
    }
}

#[test]
fn rest_builder_order_preserves_duration_and_participation() {
    for preset in ["entrained()", "steady()", "flow()"] {
        let words = ["sustain()", preset, "adaptive_duration()", "rest()"];
        for a in 0..4 {
            for b in 0..4 {
                for c in 0..4 {
                    for d in 0..4 {
                        if [a, b, c, d]
                            .iter()
                            .enumerate()
                            .any(|(i, n)| [a, b, c, d][..i].contains(n))
                        {
                            continue;
                        }
                        let chain = [words[a], words[b], words[c], words[d]].join(".");
                        let spec = first_spawn_spec_for_script(&format!(
                            "create(harmonic().{chain}.duration_range(0.2,0.2),1);"
                        ));
                        assert!(
                            spec.control.phonation.spec.is_sustained_participation(),
                            "{chain}"
                        );
                        assert_eq!(spec.control.phonation.rest_periods, Some(1.));
                        assert!(
                            matches!(spec.control.phonation.finite_duration,Some(DurationSpec::Field(f)) if f.hold_min_theta==0.2 && f.hold_max_theta==0.2)
                        );
                        assert_eq!(spec.control.pitch.pitch_apply_mode, PitchApplyMode::Glide);
                    }
                }
            }
        }
        for chain in [
            format!("cycles(2).sustain().{preset}.rest(0.5)"),
            format!("rest(0.5).sustain().{preset}.cycles(2)"),
        ] {
            let spec = first_spawn_spec_for_script(&format!("create(sine().{chain},1);"));
            assert!(spec.control.phonation.spec.is_sustained_participation());
            assert_eq!(
                spec.control.phonation.finite_duration,
                Some(DurationSpec::Gates(2))
            );
        }
    }
}

#[test]
fn articulation_persistence_builder_order_and_common_creation_point() {
    fn permutations(parts: &[&str]) -> Vec<String> {
        if parts.is_empty() {
            return vec![String::new()];
        }
        let mut result = Vec::new();
        for (index, part) in parts.iter().enumerate() {
            let remaining: Vec<_> = parts
                .iter()
                .enumerate()
                .filter_map(|(i, p)| (i != index).then_some(*p))
                .collect();
            for tail in permutations(&remaining) {
                result.push(format!(".{part}{tail}"));
            }
        }
        result
    }
    for timing in ["entrained()", "steady()", "flow()"] {
        for chain in permutations(&[
            "sustain()",
            timing,
            "pitch_apply_mode(\"rearticulate\")",
            "rearticulate_above(50)",
            "articulation_persistence(0.75)",
        ]) {
            for script in [
                format!("create(sine(){chain},1);"),
                format!("place(sine(){chain},line(330.0,330.0));"),
            ] {
                let spec = first_spawn_spec_for_script(&script);
                assert!(spec.control.supports_articulation_persistence(), "{script}");
                assert_eq!(spec.control.phonation.articulation_persistence, 0.75);
            }
        }
    }
}

#[test]
fn articulation_persistence_respawn_during_suspension_can_reenter_scope() {
    use crate::life::phonation_engine::PhonationClock;
    use crate::life::shape_policy::Shape;
    use crate::life::voice::PhonationBatch;
    for suspend in ["p.pitch_apply_mode(\"glide\");", "p.rearticulate_above(0);"] {
        let (scenario, _) = run_script_with_seed(
            &format!(
                "let p=create(sine().sustain().entrained().pitch_apply_mode(\"rearticulate\").rearticulate_above(50).articulation_persistence(0.75).respawn_random().respawn_background_death_rate(10000.0),1); {suspend} flush();"
            ),
            21,
        );
        let tb = Timebase {
            fs: 48000.,
            hop: 257,
        };
        let mut community = Community::new(tb);
        community.set_seed(scenario.seed);
        let landscape = Landscape::default();
        for action in scenario.events.iter().flat_map(|event| &event.actions) {
            if matches!(
                action,
                Action::Spawn { .. }
                    | Action::SetRespawnPolicy { .. }
                    | Action::UpdatePopulation { .. }
            ) {
                community.apply_action(action.clone(), &landscape);
            }
        }
        let parent = community.voices[0].id();
        let PhonationClock::Participation(clock) = &mut community.voices[0].phonation_engine.clock
        else {
            panic!()
        };
        clock.shape.as_mut().unwrap().state = Shape::Connect;
        community.advance(257, 0, 257. / 48000., &landscape);
        community.cleanup_dead(0, 257. / 48000., false, &landscape);
        assert!(community.voices.iter().any(|v| v.id() != parent));
        for child in community.voices.iter_mut().filter(|v| v.id() != parent) {
            assert_eq!(
                child.effective_control.phonation.articulation_persistence,
                0.75
            );
            assert!(!child.effective_control.supports_articulation_persistence());
            let PhonationClock::Participation(clock) = &child.phonation_engine.clock else {
                panic!()
            };
            let shape = clock
                .shape
                .as_ref()
                .expect("inherited persistence must survive temporary suspension");
            assert_eq!(shape.state, Shape::Base);
            assert!(!shape.enabled && !shape.veto_plan && !shape.has_tone);
            child
                .apply_patch(&crate::scenario::control::ControlUpdate {
                    pitch_apply_mode: Some(crate::scenario::control::PitchApplyMode::Rearticulate),
                    rearticulate_above_cents: Some(50.),
                    ..Default::default()
                })
                .unwrap();
            let mut out = PhonationBatch::default();
            child.prepare_phonation_batch(&mut out);
            child.tick_phonation_into_with_arrival(
                &tb,
                257,
                &landscape.rhythm,
                None,
                0.,
                1.,
                0.8,
                None,
                None,
                &mut out,
            );
            let PhonationClock::Participation(clock) = &child.phonation_engine.clock else {
                panic!()
            };
            assert!(clock.shape.as_ref().unwrap().enabled);
            assert_eq!(out.tones.len(), 1);
        }
    }
}

#[test]
fn tie_builder_order_values_scope_and_no_live_setter() {
    for timing in ["entrained()", "steady()", "flow()"] {
        for mode in ["gate_snap", "rearticulate"] {
            for chain in [
                format!("tie(0.5).sustain().{timing}.pitch_apply_mode(\"{mode}\")"),
                format!("sustain().{timing}.pitch_apply_mode(\"{mode}\").tie(0.5)"),
            ] {
                for script in [
                    format!("create(sine().{chain},1);"),
                    format!("place(sine().{chain},line(330.0,330.0));"),
                ] {
                    let spec = first_spawn_spec_for_script(&script);
                    assert_eq!(spec.control.phonation.tie, 0.5, "{script}");
                    assert!(spec.control.supports_tie());
                }
            }
        }
    }
    for (value, expected) in [
        (0., 0.),
        (1., 1.),
        (-1., 0.),
        (2., 1.),
        (f32::NAN, 0.),
        (f32::INFINITY, 1.),
        (f32::NEG_INFINITY, 0.),
    ] {
        let ctx = Arc::new(Mutex::new(ScriptContext::with_seed(21)));
        let mut engine = ScriptHost::create_engine(ctx.clone());
        engine.register_fn("test_value", move || value as rhai::FLOAT);
        let _ = engine.eval::<Dynamic>("create(sine().sustain().entrained().pitch_apply_mode(\"gate_snap\").tie(test_value()),1);").unwrap();
        assert_eq!(
            ctx.lock()
                .unwrap()
                .populations
                .values()
                .next()
                .unwrap()
                .spec
                .control
                .phonation
                .tie,
            expected
        );
    }
    for chain in [
        "sustain()",
        "sustain().entrained()",
        "pulse(2).cycles(1)",
        "sustain().entrained().anchor().pitch_apply_mode(\"gate_snap\")",
    ] {
        assert_eq!(
            first_spawn_spec_for_script(&format!("create(sine().{chain}.tie(1),1);"))
                .control
                .phonation
                .tie,
            0.0
        );
    }
    assert!(run_script_err("let p=create(sine().sustain().entrained().pitch_apply_mode(\"gate_snap\").tie(1),1); p.tie(0.5);").to_string().contains("tie"));
}

#[test]
fn ignored_controls_are_collected_with_their_input_positions() {
    for (expression, label) in [
        ("sine().tie(0.5)", "tie()"),
        (
            "harmonic().rhythm_reward(1, \"attack_phase_match\").sustain()",
            "rhythm_reward()",
        ),
        ("sine().tie(0)", "tie()"),
        (
            "sine().articulation_persistence(0)",
            "articulation_persistence()",
        ),
        ("sine().rearticulate_above(0)", "rearticulate_above()"),
        (
            "harmonic().sustain().pitch_apply_mode(\"rearticulate\").rhythm_reward(1, \"attack_phase_match\")",
            "rhythm_reward()",
        ),
        (
            "harmonic().rhythm_reward(1, \"attack_phase_match\").repeat().pulse(2)",
            "rhythm_reward()",
        ),
        (
            "sine().articulation_persistence(0.5)",
            "articulation_persistence()",
        ),
        ("sine().intrinsic_rate(2)", "intrinsic_rate()"),
        ("sine().rearticulate_above(50)", "rearticulate_above()"),
        ("harmonic().brain(\"drone\").endurance(2)", "endurance()"),
        ("harmonic().recovery(2).brain(\"seq\")", "recovery()"),
        (
            "harmonic().repeat().pulse(2).rhythm_reward(2, \"attack_phase_match\")",
            "rhythm_reward()",
        ),
    ] {
        let (_, warnings) = run_script(&format!(
            "\nlet spec = {expression};\nplace(spec, at(220));"
        ));
        let warnings = warnings.lock().unwrap();
        let warning = warnings
            .iter()
            .find(|warning| warning.message.contains(label))
            .unwrap_or_else(|| panic!("missing {label}: {warnings:?}"));
        assert_eq!(warning.position.line(), Some(2), "{expression}");
    }
    for expression in [
        "at(220).peak()",
        "at(220).tension(0.5)",
        "dissonance(220, 330).tension(0.5)",
        "at(220).density()",
        "at(220).range(1, 2)",
        "at(220).spacing(1)",
        "harmonic_modes().range(1, 2)",
        "harmonic_modes().spacing(1)",
        "harmonic_modes().gamma(2)",
    ] {
        let (_, warnings) = run_script(&format!("\nlet unused = {expression};"));
        let warnings = warnings.lock().unwrap();
        assert_eq!(warnings.len(), 1, "{expression}: {warnings:?}");
        assert_eq!(warnings[0].position.line(), Some(2));
    }
    let (_, warnings) = run_script(
        "place(harmonic().sustain().rhythm_reward(1, \"attack_phase_match\"), at(220));",
    );
    assert!(warnings.lock().unwrap().is_empty());
}

#[test]
fn invalid_enum_names_stop_before_writes_and_show_one_position() {
    for (function, canonical, aliases) in [
        (
            "pitch_apply_mode",
            "glide",
            vec!["snap", "gatesnap", "gliss", "glissando"],
        ),
        (
            "pitch_core",
            "hill_climb",
            vec!["hill", "peak", "hillclimb"],
        ),
        (
            "move_cost_time_scale",
            "proposal_interval",
            vec!["proposal", "integration", "integration_window"],
        ),
        ("viability_scope", "environment", vec!["off"]),
        ("brain", "drone", vec!["dronee"]),
        ("rhythm_role", "beat", vec!["beats"]),
    ] {
        for bad in aliases.into_iter().map(str::to_string).chain([
            canonical.to_uppercase(),
            format!(" {canonical}"),
            "unknown".into(),
        ]) {
            let error = run_script_err(&format!("\nlet spec = harmonic().{function}(\"{bad}\");"));
            assert_eq!(error.position.unwrap().line(), Some(2), "{error}");
            assert_eq!(error.to_string().matches("line 2").count(), 1, "{error}");
            assert!(error.message.contains(function));
        }
    }
    for call in [
        "rhythm_reward(1, \"phase_match\")",
        "rhythm_reward(1, \"off\")",
        "rhythm_reward(1, \"disabled\")",
    ] {
        assert!(
            run_script_err(&format!("harmonic().{call};"))
                .message
                .contains("rhythm_reward")
        );
    }
    for call in [
        "set_pitch_objective(\"negative\")",
        "set_pitch_objective(\"positive\")",
        "control_update_mode(\"snapshot\")",
        "modal_table(\"wineglass\")",
        "temporal_mode(\"OBSERVE\")",
    ] {
        assert!(run_script_err(&format!("\n{call};")).position.is_some());
    }
    for method in ["pitch_apply_mode", "move_cost_time_scale"] {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        assert!(
            engine
                .eval::<Dynamic>(&format!(
                    "let p = place(harmonic(), at(220)); p.{method}(\"unknown\");"
                ))
                .is_err()
        );
        let ctx = ctx.lock().unwrap();
        let population = ctx.populations.values().next().unwrap();
        assert_eq!(population.pending_patch, ControlUpdate::default());
    }
}

#[test]
fn removed_loo_mode_is_not_a_script_control() {
    for source in [
        r#"harmonic().leave_self_out_mode("approx");"#,
        r#"let p = place(harmonic(), at(220)); p.leave_self_out_mode("exact");"#,
    ] {
        assert!(
            run_script_err(source)
                .message
                .contains("leave_self_out_mode")
        );
    }
}

#[test]
fn flora_specs_emit_placement_actions_and_route_buses() {
    let (scenario, _) = run_script(
        r#"
        let moss = flora().amp(0.1).rate(1).renewal(10).fade(0.25)
            .tone("resonant").residual(false).send(presentation_bus);
        place(moss, consonance(110).range(1, 16));
        place(flora().residual(true).send(habitat_bus), random(220, 880));
        place(flora().send(habitat_bus | presentation_bus), consonance(110, 880));
        place(flora(), random(110, 880));
        wait(0.5);
        "#,
    );
    let placed: Vec<_> = action_times(&scenario)
        .into_iter()
        .filter_map(|(time, action)| match action {
            Action::PlaceFlora {
                flora_id,
                spec,
                extent_hz,
                seeding,
            } => Some((time, *flora_id, spec, *extent_hz, seeding)),
            _ => None,
        })
        .collect();
    assert_eq!(placed.len(), 4);
    let (time, id, spec, extent, seeding) = placed[0];
    assert_eq!((time, id, extent), (0.0, 1, (110.0, 1760.0)));
    assert!(matches!(seeding, FloraSeeding::Consonance));
    assert_eq!(
        (spec.amp, spec.rate, spec.renewal_sec, spec.fade_sec),
        (0.1, 1.0, 10.0, 0.25)
    );
    assert_eq!(spec.tone, FloraTone::Resonant);
    assert!(!spec.residual);
    assert!(!spec.routing.to_habitat && spec.routing.to_presentation);
    assert_eq!(placed[1].1, 2);
    assert_eq!(placed[1].3, (220.0, 880.0));
    assert!(matches!(placed[1].4, FloraSeeding::Uniform));
    assert!(placed[1].2.residual);
    assert!(placed[1].2.routing.to_habitat && !placed[1].2.routing.to_presentation);
    for (_, _, spec, _, _) in &placed[2..] {
        assert!(spec.routing.to_habitat && spec.routing.to_presentation);
        assert_eq!(
            (spec.amp, spec.rate, spec.renewal_sec, spec.fade_sec),
            (0.03, 8.0, 10.0, 2.0)
        );
        assert_eq!(spec.tone, FloraTone::Sine);
        assert!(!spec.residual);
    }
    assert!(
        !action_times(&scenario)
            .iter()
            .any(|(_, a)| matches!(a, Action::ReleaseFlora { .. }))
    );
    assert_eq!(scenario.duration_sec, 0.5);
}

#[test]
fn flora_accepts_value_based_placement_defaults() {
    for placement in [
        "consonance(110, 1760)",
        "consonance(110).range(1, 16)",
        "consonance(110, 1760).count(1).density().tension(0).spacing(1)",
        "random(110, 1760).count(1).density().tension(0).spacing(0)",
        "random(1, 20000)",
        "random(110, 880).tension(1)",
        "consonance(110, 880).tension(-1)",
    ] {
        let (scenario, _) = run_script(&format!("place(flora(), {placement});"));
        assert!(
            matches!(scenario.events[0].actions[0], Action::PlaceFlora { .. }),
            "{placement}"
        );
    }
}

#[test]
fn flora_rejects_each_unsupported_placement_component() {
    for (placement, message) in [
        ("at(220)", "at placement"),
        ("line(110, 880)", "line placement"),
        ("dissonance(110, 880)", "Dissonance placement target"),
        ("edge(110, 880)", "Edge placement target"),
        ("gap(110, 880)", "Gap placement target"),
        ("consonance(110, 880).count(0)", "count must be 1"),
        ("consonance(110, 880).count(-1)", "count must be 1"),
        ("random(110, 880).count(2)", "count must be 1"),
        ("consonance(110, 880).peak()", "peak sampling"),
        ("random(110, 880).peak()", "peak sampling"),
        ("consonance(110, 880).tension(0.5)", "tension must be 0"),
        ("consonance(110, 880).spacing(0)", "spacing must be 1"),
        ("random(110, 880).spacing(1)", "spacing must be 0"),
        ("consonance(110, 110.1)", "at least one Log2Space bin"),
        ("consonance(110, 110)", "at least one Log2Space bin"),
        ("random(880, 110)", "at least one Log2Space bin"),
        ("consonance(0, 880)", "field placement bounds"),
        ("random(110, 20001)", "field placement bounds"),
        ("consonance(1e100, 880)", "field placement bounds"),
    ] {
        let error = run_script_err(&format!("place(flora(), {placement});")).to_string();
        assert!(error.contains(message), "{placement}: {error}");
    }
    // Bounds are already f32 in a Placement; exercise NaN and both infinities.
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1e100] {
        for placement in ["consonance(value, 880)", "random(110, value)"] {
            let ctx = Arc::new(Mutex::new(ScriptContext::default()));
            let engine = ScriptHost::create_engine(ctx);
            let mut scope = rhai::Scope::new();
            scope.push("value", value);
            let error = engine
                .eval_with_scope::<Dynamic>(&mut scope, &format!("place(flora(), {placement});"))
                .expect_err("invalid extent")
                .to_string();
            assert!(
                error.contains("field placement bounds"),
                "{placement}: {error}"
            );
        }
    }
}

#[test]
fn flora_numeric_validation_covers_specs_and_patches() {
    for (name, min, max) in [
        ("amp", 0.0, 1.0),
        ("rate", 0.05, 8.0),
        ("renewal", 0.5, 600.0),
        ("fade", 0.0, 60.0),
    ] {
        for patch in [false, true] {
            if patch && name == "fade" {
                continue;
            }
            let script = if patch {
                format!("let m = place(flora(), random(110, 880)); m.{name}(value);")
            } else {
                format!("flora().{name}(value);")
            };
            for value in [min, max] {
                let ctx = Arc::new(Mutex::new(ScriptContext::default()));
                let engine = ScriptHost::create_engine(ctx);
                let mut scope = rhai::Scope::new();
                scope.push("value", value);
                let _ = engine
                    .eval_with_scope::<Dynamic>(&mut scope, &script)
                    .expect("inclusive boundary accepted");
            }
            for value in [
                min - 0.001,
                max + 0.001,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
                1e100,
                -1e100,
            ] {
                let ctx = Arc::new(Mutex::new(ScriptContext::default()));
                let engine = ScriptHost::create_engine(ctx);
                let mut scope = rhai::Scope::new();
                scope.push("value", value);
                let error = engine
                    .eval_with_scope::<Dynamic>(&mut scope, &script)
                    .expect_err("invalid Flora numeric argument")
                    .to_string();
                let expected = if !value.is_finite() {
                    format!("Flora {name} must be finite before and after f32 conversion")
                } else {
                    format!("Flora {name} must be within [{min}, {max}]")
                };
                assert!(
                    error.contains(&expected),
                    "{script}, value={value}: {error}"
                );
            }
            for value in [-1_i64, max as i64 + 1] {
                let script = script.replace("value", &value.to_string());
                let error = run_script_err(&script).to_string();
                assert!(
                    error.contains(&format!("Flora {name} must be within")),
                    "{script}: {error}"
                );
            }
        }
    }
    for tone in ["noise", "Sine", " sine", ""] {
        let error = run_script_err(&format!("flora().tone(\"{tone}\");")).to_string();
        assert!(
            error.contains("Flora tone expects sine or resonant"),
            "{error}"
        );
    }
    run_script("flora().amp(1).rate(8).renewal(600).fade(60).tone(\"sine\");");
}

#[test]
fn flora_patches_are_pending_until_wait_flush_or_finish() {
    for boundary in ["wait(0.5);", "flush();", ""] {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx.clone());
        let mut scope = rhai::Scope::new();
        let _ = engine
            .eval_with_scope::<Dynamic>(
                &mut scope,
                r#"
            let m = place(flora(), random(110, 880));
            wait(0.25);
            m.amp(0.1).amp(0.2).rate(2).renewal(10);
        "#,
            )
            .unwrap();
        assert_eq!(ctx.lock().unwrap().scenario.events.len(), 1);
        let _ = engine
            .eval_with_scope::<Dynamic>(&mut scope, boundary)
            .unwrap();
        let mut ctx = ctx.lock().unwrap();
        ctx.finish();
        let updates: Vec<_> = action_times(&ctx.scenario)
            .into_iter()
            .filter_map(|(time, action)| match action {
                Action::UpdateFlora { flora_id, patch } => Some((time, *flora_id, patch)),
                _ => None,
            })
            .collect();
        assert_eq!(updates.len(), 1);
        assert_eq!((updates[0].0, updates[0].1), (0.25, 1));
        let patch = updates[0].2;
        assert_eq!(
            (patch.amp, patch.rate, patch.renewal_sec),
            (Some(0.2), Some(2.0), Some(10.0))
        );
    }
}

#[test]
fn flora_pending_patch_precedes_release_in_one_event_and_extends_finish() {
    for boundary in ["wait(0.1);", "flush();", ""] {
        let (scenario, _) = run_script(&format!(
            r#"
            let m = place(flora().fade(3), consonance(110, 880));
            wait(2);
            m.amp(0.015);
            release(m);
            {boundary}
        "#
        ));
        let event = scenario
            .events
            .iter()
            .find(|e| {
                e.actions
                    .iter()
                    .any(|a| matches!(a, Action::ReleaseFlora { .. }))
            })
            .unwrap();
        assert_eq!(event.time, 2.0);
        assert!(matches!(event.actions.as_slice(), [
            Action::UpdateFlora { flora_id: 1, patch: FloraPatch { amp: Some(amp), .. } },
            Action::ReleaseFlora { flora_id: 1, fade_sec: 3.0 }
        ] if *amp == 0.015));
        assert_eq!(scenario.duration_sec, 5.0);
        assert!(matches!(
            scenario.events.last().unwrap().actions.as_slice(),
            [Action::Finish]
        ));
        assert_eq!(scenario.events.last().unwrap().time, 5.0);
    }
}

#[test]
fn flora_nested_scopes_release_inner_at_inner_end_with_patches_first() {
    let (scenario, _) = run_script(
        r#"
        section("outer", || {
            let outer = place(flora().fade(0.5), random(110, 880));
            let voices = place(sine(), at(220));
            play(|| {
                let inner = place(flora().fade(0.25), consonance(110, 880));
                wait(1);
                inner.amp(0.1);
            });
            wait(1);
            voices.amp(0.05);
            outer.rate(1);
        });
    "#,
    );
    let releases: Vec<_> = scenario
        .events
        .iter()
        .filter(|e| {
            e.actions
                .iter()
                .any(|a| matches!(a, Action::ReleaseFlora { .. }))
        })
        .collect();
    assert_eq!(releases.len(), 2);
    assert_eq!(releases[0].time, 1.0);
    assert!(matches!(
        releases[0].actions.as_slice(),
        [
            Action::UpdateFlora { flora_id: 2, .. },
            Action::ReleaseFlora {
                flora_id: 2,
                fade_sec: 0.25
            }
        ]
    ));
    assert_eq!(releases[1].time, 2.0);
    assert!(matches!(
        releases[1].actions.as_slice(),
        [
            Action::UpdatePopulation { .. },
            Action::UpdateFlora { flora_id: 1, .. },
            Action::ReleasePopulation { .. },
            Action::ReleaseFlora {
                flora_id: 1,
                fade_sec: 0.5
            }
        ]
    ));
    assert_eq!(scenario.duration_sec, 2.5);
}

#[test]
fn flora_all_scoped_verbs_release_owned_flora() {
    for script in [
        "section(\"flora\", || { let m = place(flora(), random(110, 880)); m.amp(0.1); release(m); });",
        "play(|| { place(flora(), random(110, 880)); });",
        "play(|a| { place(flora(), random(110, 880)); }, 1);",
        "play(|a, b| { place(flora(), random(110, 880)); }, 1, 2);",
        "play(|a, b, c| { place(flora(), random(110, 880)); }, 1, 2, 3);",
        "play(|a, b, c, d| { place(flora(), random(110, 880)); }, [1, 2, 3, 4]);",
        "parallel([|| { place(flora(), random(110, 880)); }, || { place(flora(), consonance(110, 880)); }]);",
    ] {
        let (scenario, _) = run_script(script);
        let actions = action_times(&scenario);
        let placed = actions
            .iter()
            .filter(|(_, a)| matches!(a, Action::PlaceFlora { .. }))
            .count();
        let released = actions
            .iter()
            .filter(|(_, a)| matches!(a, Action::ReleaseFlora { .. }))
            .count();
        assert_eq!(placed, released, "{script}");
    }
}

#[test]
fn flora_warnings_are_collected_with_call_positions() {
    let (scenario, warnings) = run_script(
        r#"
        let m = place(flora(), random(110, 880));
        release(m); release(m); flush(); release(m); m.amp(0.5);
        let scoped = ();
        play(|| { scoped = place(flora(), random(110, 880)); });
        release(scoped); scoped.renewal(5);
    "#,
    );
    let warnings = warnings.lock().unwrap();
    for message in [
        "release already pending",
        "release ignored for inactive Flora 1",
        "amp ignored for inactive Flora 1",
        "release ignored for inactive Flora 2",
        "renewal ignored for inactive Flora 2",
    ] {
        let warning = warnings
            .iter()
            .find(|w| w.message.contains(message))
            .expect(message);
        assert_ne!(warning.position, Position::NONE, "{warning:?}");
    }
    assert_eq!(
        action_times(&scenario)
            .iter()
            .filter(|(_, a)| matches!(a, Action::ReleaseFlora { .. }))
            .count(),
        2
    );
    assert!(
        !action_times(&scenario)
            .iter()
            .any(|(_, a)| matches!(a, Action::UpdateFlora { .. }))
    );
}

#[test]
fn flora_errors_preserve_native_call_positions() {
    for source in [
        "\nflora().amp(1.00000001);",
        "\nflora().tone(\"Sine\");",
        "\nplace(flora(), at(220));",
        "\nlet m = place(flora(), random(110, 880)); m.rate(8.00000001);",
    ] {
        let error = run_script_err(source);
        assert!(
            error.position.is_some_and(|p| p != Position::NONE),
            "{error:?}"
        );
    }
}

#[test]
fn flora_metric_is_opt_in_initial_only_and_independent_of_population_metric() {
    for (chain, expected) in [("", false), (".metric()", true)] {
        let (scenario, _) = run_script(&format!("place(flora(){chain}, random(110,880));"));
        let spec = scenario
            .events
            .iter()
            .flat_map(|e| &e.actions)
            .find_map(|a| match a {
                Action::PlaceFlora { spec, .. } => Some(spec),
                _ => None,
            })
            .unwrap();
        assert_eq!(spec.metric, expected);
    }
    assert!(
        run_script_err("let f=place(flora(),random(110,880)); f.metric();")
            .to_string()
            .contains("metric")
    );
}

#[test]
fn flora_metric_uses_scene_tempo_instead_of_local_rate_overrides() {
    for script in [
        "place(flora().metric().rate(4), random(110, 1760));",
        "place(flora().renewal(20).metric(), random(110, 1760));",
        "let f = place(flora().metric(), random(110, 1760)); f.rate(4);",
        "let f = place(flora().metric(), random(110, 1760)); f.renewal(20);",
    ] {
        let ctx = Arc::new(Mutex::new(ScriptContext::default()));
        let engine = ScriptHost::create_engine(ctx);
        let error = engine
            .eval::<Dynamic>(script)
            .expect_err("metric time override");
        assert!(error.to_string().contains("temporal_basin"), "{error}");
    }
    let scenario =
        run_script("temporal_basin(2, 4); place(flora().metric(), random(110, 1760)); wait(1);").0;
    assert_eq!(scenario.meter_shaping.basin_hz, Some((2.0, 4.0)));
}
