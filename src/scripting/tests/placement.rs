use super::*;

#[test]
fn placement_count_must_be_positive() {
    for count in [0, -1] {
        let err = run_script_err(&format!("place(sine(), at(220.0).count({count}));"));
        assert!(
            err.message.contains("population count must be >= 1"),
            "unexpected error for count {count}: {}",
            err.message
        );
    }
}

#[test]
fn field_placement_bounds_must_fit_sound_body_frequency_limits() {
    for placement in [
        "consonance(30000.0, 31000.0)",
        "dissonance(0.5, 100.0)",
        "edge(-100.0, 100.0)",
        "gap(100.0, 0.0)",
        "random(100.0, 20001.0)",
        "consonance(10000.0).range(1.0, 3.0)",
        "consonance(1e100, 100.0)",
    ] {
        let err = run_script_err(&format!("place(sine(), {placement});"));
        assert!(
            err.message
                .contains("field placement bounds must be finite"),
            "{placement}: {}",
            err.message
        );
    }
    let err = run_script_err(
        "place(sine().respawn_random().respawn_settle(consonance(30000.0, 31000.0)), at(220.0));",
    );
    assert!(
        err.message
            .contains("field placement bounds must be finite")
    );
    run_script("place(sine(), consonance(20000.0, 1.0));");
}

#[test]
fn population_spec_is_reusable_across_placements() {
    let (scenario, _warnings) = run_script(
        r#"
            let spec = sine().amp(0.2);
            place(spec, at(220.0));
            place(spec, at(330.0));
        "#,
    );
    let spawns = scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
        .filter(|action| matches!(action, Action::Spawn { .. }))
        .count();
    assert_eq!(spawns, 2);
}

#[test]
fn field_can_be_user_variable_when_using_habitat_bus() {
    let (scenario, _warnings) = run_script(
        r#"
            let field = place(harmonic().amp(0.04).send(habitat_bus), at(220.0));
            field.freq(330.0);
            flush();
        "#,
    );
    let (control, patch_freq) = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some((spec.control.clone(), None)),
            _ => None,
        })
        .expect("spawn action");
    let patch_freq = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::UpdatePopulation { patch, .. } => patch.freq,
            _ => None,
        })
        .or(patch_freq);

    assert!(control.body.routing.to_habitat);
    assert!(!control.body.routing.to_presentation);
    assert!((control.pitch.freq - 220.0).abs() <= 1e-6);
    assert_eq!(patch_freq, Some(330.0));
}

#[test]
fn population_freq_patch_preserves_founder_placement() {
    let (scenario, _warnings) = run_script(
        r#"
            let population = place(sine(), consonance(220.0).peak().count(4));
            population.freq(330.0);
            flush();
        "#,
    );
    let strategy = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { strategy, .. } => Some(strategy.clone()),
            _ => None,
        })
        .expect("spawn action");
    assert!(matches!(
        strategy,
        Some(SpawnStrategy::Field {
            target: FieldTarget::Consonance,
            ..
        })
    ));
    let patch_freq = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::UpdatePopulation { patch, .. } => patch.freq,
            _ => None,
        });
    assert_eq!(patch_freq, Some(330.0));
}

#[test]
fn at_placement_respects_final_pitch_intent() {
    for (chain, mode) in [
        ("", PitchMode::Lock),
        (".anchor()", PitchMode::Lock),
        (".freq(110.0)", PitchMode::Lock),
        (".seek_consonance()", PitchMode::Free),
        (".seek_consonance().freq(110.0)", PitchMode::Lock),
        (".freq(110.0).seek_consonance()", PitchMode::Free),
        (".seek_consonance().anchor()", PitchMode::Lock),
        (".anchor().seek_consonance()", PitchMode::Free),
    ] {
        let (scenario, _) = run_script(&format!("place(sine(){chain}, at(330.0).count(3));"));
        let mut community = Community::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        let landscape = Landscape::default();
        for action in scenario.events.iter().flat_map(|event| &event.actions) {
            if let Action::Spawn {
                ids,
                spec,
                strategy,
                ..
            } = action
            {
                assert_eq!(ids.len(), 3, "{chain}");
                assert_eq!(spec.control.pitch.mode, mode, "{chain}");
                if mode == PitchMode::Free {
                    assert!(
                        matches!(
                            strategy,
                            Some(SpawnStrategy::Linear {
                                start_freq: 330.0,
                                end_freq: 330.0
                            })
                        ),
                        "{chain}"
                    );
                } else {
                    assert!(strategy.is_none(), "{chain}");
                    assert_eq!(spec.control.pitch.freq, 330.0, "{chain}");
                }
                community.apply_action(action.clone(), &landscape);
            }
        }
        assert_eq!(community.voices.len(), 3);
        for voice in &community.voices {
            assert_eq!(voice.base_control.pitch.mode, mode, "{chain}");
            assert_eq!(voice.effective_control.pitch.mode, mode, "{chain}");
            assert_eq!(voice.body.base_freq_hz(), 330.0, "{chain}");
        }
    }
}

#[test]
fn rearticulate_final_placement_validates_both_builder_orders() {
    for timing in [
        "repeat().pulse(2.0)",
        "metric()",
        "entrained()",
        "cycles(1)",
    ] {
        for chain in [
            format!("sine().seek_consonance().pitch_apply_mode(\"rearticulate\").{timing}"),
            format!("sine().{timing}.seek_consonance().pitch_apply_mode(\"rearticulate\")"),
        ] {
            let err = run_script_err(&format!("place({chain}, line(330.0, 330.0).count(1));"));
            let message = err.to_string();
            assert!(message.contains("Gated"), "{message}");
            assert!(message.contains("rearticulate"));
            assert!(message.contains("sustain()"));
            assert!(message.contains("gate_snap"));
            assert!(message.contains("glide"));
        }
    }
    for script in [
        "place(sine().seek_consonance().pitch_apply_mode(\"rearticulate\").pulse(2.0).sustain(), line(330.0, 330.0));",
        "place(sine().sustain().pitch_apply_mode(\"rearticulate\").seek_consonance(), line(330.0, 330.0));",
        "place(sine().anchor().pulse(2.0).pitch_apply_mode(\"rearticulate\"), at(330.0));",
        "let p = place(sine().sustain().seek_consonance(), line(330.0, 330.0)); p.pitch_apply_mode(\"rearticulate\"); wait(1.0); p.pitch_apply_mode(\"glide\"); wait(1.0);",
    ] {
        run_script(script);
    }
}

#[test]
fn rest_invalid_placement_and_live_writes_are_transactional() {
    for chain in [
        "sustain().entrained().rest()",
        "sustain().entrained().adaptive_duration().while_alive().rest()",
        "adaptive_duration().sustain().rest()",
    ] {
        let ctx = Arc::new(Mutex::new(ScriptContext::with_seed(21)));
        let engine = ScriptHost::create_engine(ctx.clone());
        let err = engine
            .eval::<Dynamic>(&format!("create(sine().{chain},1);"))
            .expect_err("invalid rest");
        assert!(err.to_string().contains("rest()"));
        let state = ctx.lock().unwrap();
        assert!(state.populations.is_empty());
        assert_eq!(state.next_population_id, 1);
    }
    let ctx = Arc::new(Mutex::new(ScriptContext::with_seed(21)));
    let engine = ScriptHost::create_engine(ctx.clone());
    let mut scope = rhai::Scope::new();
    let _ = engine.eval_with_scope::<Dynamic>(&mut scope,"let p=create(sine().sustain().entrained().cycles(1).rest().pitch_apply_mode(\"gate_snap\"),1);").unwrap();
    let _ = engine
        .eval_with_scope::<Dynamic>(
            &mut scope,
            "p.pitch_apply_mode(\"rearticulate\"); p.rearticulate_above(50);",
        )
        .unwrap();
    let before = format!("{:?}", ctx.lock().unwrap().populations);
    for call in ["p.rest();", "p.cycles(2);", "p.duration_range(0.2,0.3);"] {
        assert!(engine.eval_with_scope::<Dynamic>(&mut scope, call).is_err());
        assert_eq!(format!("{:?}", ctx.lock().unwrap().populations), before);
    }
}
