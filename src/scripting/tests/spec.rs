use super::*;

#[test]
fn steady_is_initial_only_and_timing_presets_use_the_last_written_policy() {
    for earlier in ["steady()", "entrained()", "flow()", "metric()"] {
        for last in ["steady()", "entrained()", "flow()", "metric()"] {
            let combined =
                first_spawn_spec_for_script(&format!("create(sine().{earlier}.{last},1);"));
            let expected = first_spawn_spec_for_script(&format!("create(sine().{last},1);"));
            assert_eq!(
                combined.control.phonation.spec.timing,
                expected.control.phonation.spec.timing
            );
            assert_eq!(
                combined.control.phonation.tie,
                expected.control.phonation.tie
            );
        }
    }
    let mut expected = first_spawn_spec_for_script("create(sine().entrained(),1);")
        .control
        .phonation
        .spec;
    let PhonationTiming::Coupled(timing) = &mut expected.timing else {
        panic!()
    };
    timing.steady = true;
    assert_eq!(
        first_spawn_spec_for_script("create(sine().steady(),1);")
            .control
            .phonation
            .spec,
        expected
    );
    for chain in ["steady().entrainment(0.3)", "entrainment(0.3).steady()"] {
        let spec = first_spawn_spec_for_script(&format!("create(sine().{chain},1);"));
        assert!(
            matches!(spec.control.phonation.spec.timing, PhonationTiming::Coupled(t) if t.steady && t.coupling == 0.3 && t.flow_depth == 0.0)
        );
    }
    assert!(
        run_script_err("let p=create(sine().steady(),1); p.steady();")
            .to_string()
            .contains("steady")
    );
}

#[test]
fn steady_sustain_combines_pitch_shape_and_rest_controls() {
    for pitch in ["glide", "gate_snap", "rearticulate"] {
        for chain in ["sustain().steady()", "steady().sustain()"] {
            let (scenario, _) = run_script(&format!(
                "create(sine().{chain}.seek_consonance().pitch_apply_mode(\"{pitch}\").rearticulate_above(50).articulation_persistence(0.75).tie(0.5).cycles(1).rest(0.5),1);"
            ));
            let spec = scenario
                .events
                .iter()
                .flat_map(|e| &e.actions)
                .find_map(|a| match a {
                    Action::Spawn { spec, .. } => Some(spec),
                    _ => None,
                })
                .unwrap();
            assert!(spec.control.phonation.spec.is_sustained_participation());
            assert_eq!(spec.control.phonation.rest_periods, Some(0.5));
            assert_eq!(
                spec.control.phonation.tie,
                if pitch == "glide" { 0.0 } else { 0.5 }
            );
            assert_eq!(
                spec.control.phonation.articulation_persistence,
                if pitch == "rearticulate" { 0.75 } else { 0.0 }
            );
        }
    }
    assert_eq!(
        first_spawn_spec_for_script(
            "create(sine().steady().sustain().seek_consonance().pitch_apply_mode(\"gate_snap\"),1);"
        )
        .control
        .phonation
        .tie,
        0.0
    );
}

#[test]
fn intrinsic_rate_rejects_invalid_inputs_before_creating_a_population() {
    for preset in ["entrained()", "steady()", "flow()"] {
        for placement in ["create(sine().CHAIN,1);", "place(sine().CHAIN,at(330.0));"] {
            for rate in [
                -1.0,
                0.0,
                0.005,
                0.00999999999999,
                200.00000000001,
                201.0,
                FLOAT::NAN,
                FLOAT::INFINITY,
                FLOAT::NEG_INFINITY,
            ] {
                let ctx = Arc::new(Mutex::new(ScriptContext::with_seed(21)));
                let mut engine = ScriptHost::create_engine(ctx.clone());
                engine.register_fn("test_rate", move || rate);
                let script =
                    placement.replace("CHAIN", &format!("intrinsic_rate(test_rate()).{preset}"));
                let error = engine.eval::<Dynamic>(&script).unwrap_err().to_string();
                assert!(error.contains("intrinsic_rate(hz)"), "{error}");
                assert!(error.contains("0.01..=200 Hz"), "{error}");
                assert!(error.contains("execution protection limits"), "{error}");
                assert!(
                    error.contains("not a musically recommended range"),
                    "{error}"
                );
                let context = ctx.lock().unwrap();
                assert!(context.populations.is_empty());
                assert!(context.scenario.events.is_empty());
            }
        }
        for value in ["0.01", "1", "2.0", "200"] {
            let spec = first_spawn_spec_for_script(&format!(
                "create(sine().{preset}.intrinsic_rate({value}),1);"
            ));
            let PhonationTiming::Coupled(timing) = spec.control.phonation.spec.timing else {
                panic!()
            };
            assert_eq!(timing.base_rate_hz, value.parse::<f32>().unwrap());
        }
        for value in ["0", "-1", "201"] {
            assert!(
                run_script_err(&format!(
                    "create(sine().{preset}.intrinsic_rate({value}),1);"
                ))
                .to_string()
                .contains("intrinsic_rate(hz)")
            );
        }
    }
}

#[test]
fn intrinsic_rate_unsupported_presets_ignore_even_invalid_values_without_a_live_setter() {
    for chain in [
        "once()",
        "repeat()",
        "pulse(1.7)",
        "metric()",
        "entrainment(0.5)",
    ] {
        let plain = first_spawn_spec_for_script(&format!(
            "create(sine().{chain},1); temporal_basin(2.8,3.4);"
        ));
        for value in ["0.0", "4.0"] {
            let ignored = first_spawn_spec_for_script(&format!(
                "create(sine().intrinsic_rate({value}).{chain},1); temporal_basin(2.8,3.4);"
            ));
            assert_eq!(ignored.control.phonation.spec, plain.control.phonation.spec);
        }
    }
    let error = run_script_err(
        "let p=create(sine().entrained().intrinsic_rate(4),1); p.intrinsic_rate(2);",
    );
    assert!(error.to_string().contains("intrinsic_rate"));
}

#[test]
fn intrinsic_rate_omission_matches_explicit_existing_default_and_keeps_rhythm_freq_distinct() {
    for preset in ["entrained()", "steady()", "flow()"] {
        let plain = first_spawn_spec_for_script(&format!("create(sine().{preset},1);"));
        let explicit =
            first_spawn_spec_for_script(&format!("create(sine().{preset}.intrinsic_rate(2),1);"));
        assert_eq!(format!("{plain:?}"), format!("{explicit:?}"));
        let spec = first_spawn_spec_for_script(&format!(
            "create(sine().{preset}.rhythm_freq(1.3).intrinsic_rate(4),1);"
        ));
        let PhonationTiming::Coupled(timing) = spec.control.phonation.spec.timing else {
            panic!()
        };
        assert_eq!(timing.base_rate_hz, 4.0);
        assert!(matches!(spec.articulation,
            ArticulationCoreConfig::Entrain { rhythm_freq: Some(rate), .. } if rate == 1.3));
    }
}

#[test]
fn intrinsic_rate_respawn_reads_template_and_drops_parent_performance_state() {
    fn rebuild_engine(
        voice: &mut crate::life::voice::Voice,
        spec: &crate::scenario::PhonationSpec,
    ) {
        let shape = voice
            .phonation_engine
            .clock
            .shape_slot()
            .and_then(Option::take);
        voice.phonation_engine =
            crate::life::phonation_engine::PhonationEngine::from_spec(spec, 21);
        if let Some(slot) = voice.phonation_engine.clock.shape_slot() {
            *slot = shape;
        }
    }
    use crate::life::phonation_engine::PhonationClock;
    use crate::life::shape_policy::Shape;
    for body in ["sine", "harmonic", "modal"] {
        for preset in ["entrained()", "steady()"] {
            let (scenario, _) = run_script_with_seed(
                &format!(
                    "create({body}().sustain().{preset}.intrinsic_rate(1.0/2.128).pitch_apply_mode(\"rearticulate\").rearticulate_above(50).articulation_persistence(0.75).respawn_random().respawn_background_death_rate(10000.0),1);"
                ),
                21,
            );
            let mut community = Community::new(Timebase {
                fs: 48000.,
                hop: 257,
            });
            community.set_seed(scenario.seed);
            let landscape = Landscape::default();
            for action in scenario.events.iter().flat_map(|event| &event.actions) {
                if matches!(
                    action,
                    Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
                ) {
                    community.apply_action(action.clone(), &landscape);
                }
            }
            let parent = community.voices[0].id();
            let period = community.voices[0]
                .phonation_engine
                .clock
                .intrinsic_period_sec();
            assert_eq!(period, Some(1.0 / f64::from((1.0 / 2.128) as f32)));
            // Perturb the live parent's clock; descendants must still read the template.
            let mut changed = community.voices[0].effective_control.phonation.spec.clone();
            let PhonationTiming::Coupled(timing) = &mut changed.timing else {
                panic!()
            };
            timing.base_rate_hz = 1.3;
            rebuild_engine(&mut community.voices[0], &changed);
            assert_ne!(
                community.voices[0]
                    .phonation_engine
                    .clock
                    .intrinsic_period_sec(),
                period
            );
            let PhonationClock::Participation(clock) =
                &mut community.voices[0].phonation_engine.clock
            else {
                panic!()
            };
            let shape = clock.shape.as_mut().unwrap();
            shape.state = Shape::Connect;
            shape.veto_plan = true;
            shape.has_tone = true;
            community.advance(257, 0, 257. / 48000., &landscape);
            community.cleanup_dead(0, 257. / 48000., false, &landscape);
            let children: Vec<_> = community
                .voices
                .iter()
                .filter(|v| v.id() != parent)
                .collect();
            assert!(!children.is_empty());
            for child in children {
                assert_eq!(child.phonation_engine.clock.intrinsic_period_sec(), period);
                let PhonationClock::Participation(clock) = &child.phonation_engine.clock else {
                    panic!()
                };
                assert!(clock.policy.is_none());
                assert!(
                    matches!(child.effective_control.phonation.spec.timing, PhonationTiming::Coupled(t) if t.steady == (preset == "steady()"))
                );
                let shape = clock.shape.as_ref().unwrap();
                assert_eq!(shape.state, Shape::Base);
                assert!(!shape.veto_plan && !shape.has_tone);
            }
        }
    }
}

#[test]
fn parallel_advances_to_max_child_end() {
    let (scenario, _warnings) = run_script(
        r#"
            parallel([
                || { create(sine(), 1); wait(0.5); },
                || { create(sine(), 1); wait(1.0); }
            ]);
        "#,
    );
    let finish_time = scenario
        .events
        .iter()
        .find(|ev| ev.actions.iter().any(|a| matches!(a, Action::Finish)))
        .map(|ev| ev.time);
    let mut release_tail: f32 = 0.0;
    for event in &scenario.events {
        for action in &event.actions {
            if let Action::ReleasePopulation { fade_sec, .. } = action {
                release_tail = release_tail.max(event.time + fade_sec);
            }
        }
    }
    let expected = release_tail.max(1.0);
    assert!(matches!(finish_time, Some(t) if (t - expected).abs() <= 1e-6));
}

#[test]
fn respawn_capacity_cannot_be_below_founder_count() {
    let err = run_script_err(
        r#"
            place(
                sine().respawn_random().respawn_capacity(1),
                line(220.0, 440.0).count(2)
            );
        "#,
    );
    assert!(
        err.message
            .contains("respawn_capacity must be >= founder count"),
        "unexpected error: {}",
        err.message
    );
}

#[test]
fn freq_then_place_sets_strategy() {
    let (scenario, _warnings) = run_script(
        r#"
            place(sine().freq(330.0), consonance(220.0).peak().count(4));
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
}

#[test]
fn anchored_spec_locks_population_spawn_mode() {
    let (scenario, _warnings) = run_script(
        r#"
            place(sine().anchor(), consonance(220.0).peak().count(4));
        "#,
    );
    let (strategy, mode) = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { strategy, spec, .. } => {
                Some((strategy.clone(), spec.control.pitch.mode))
            }
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
    assert_eq!(mode, PitchMode::Lock);
}

#[test]
fn anchored_spec_survives_population_spawn() {
    let (scenario, _warnings) = run_script(
        r#"
            place(sine().anchor(), consonance(220.0).peak().count(2));
        "#,
    );
    let spawn_action = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { .. } => Some(action.clone()),
            _ => None,
        })
        .expect("spawn action");

    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    pop.apply_action(spawn_action, &landscape);

    let voice = pop.voices.first().expect("spawned");
    assert_eq!(voice.base_control.pitch.mode, PitchMode::Lock);
    assert_eq!(voice.effective_control.pitch.mode, PitchMode::Lock);
}

#[test]
fn place_preserves_default_pitch_mode() {
    let (scenario, _warnings) = run_script(
        r#"
            place(sine(), consonance(220.0).peak().count(4));
        "#,
    );
    let mode = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.control.pitch.mode),
            _ => None,
        })
        .expect("spawn action");
    assert_eq!(mode, PitchMode::Free);
}

#[test]
fn anchor_then_place_preserves_lock() {
    let (scenario, _warnings) = run_script(
        r#"
            place(sine().anchor(), consonance(220.0).peak().count(4));
        "#,
    );
    let mode = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.control.pitch.mode),
            _ => None,
        })
        .expect("spawn action");
    assert_eq!(mode, PitchMode::Lock);
}

#[test]
fn population_spec_anchor_sets_spawn_mode() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().anchor(), 1);
            flush();
        "#,
    );
    let mode = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.control.pitch.mode),
            _ => None,
        })
        .expect("spawn action");
    assert_eq!(mode, PitchMode::Lock);
}

#[test]
fn at_seek_matches_degenerate_line_with_custom_pitch_controls() {
    for count in [1, 3] {
        let chain = "harmonic().freq(110.0).seek_consonance().sustain()\
            .pitch_core(\"peak_sampler\").glide(0.12).proposal_interval(0.5)\
            .pitch_apply_mode(\"rearticulate\").rearticulate_above(75)";
        let (at, _) = run_script_with_seed(
            &format!("place({chain},at(330.0).count({count})); wait(1.0);"),
            21,
        );
        let (line, _) = run_script_with_seed(
            &format!("place({chain},line(330.0,330.0).count({count})); wait(1.0);"),
            21,
        );
        assert_eq!(format!("{at:?}"), format!("{line:?}"));
    }
}

#[test]
fn at_seek_can_adopt_a_target_from_its_initial_pitch() {
    use crate::core::landscape::Landscape;
    use crate::core::log2space::Log2Space;
    use crate::core::modulation::NeuralRhythms;

    let mut landscape = Landscape::new(Log2Space::new(55.0, 4000.0, 64));
    landscape.subjective_intensity.fill(1.0);
    let target = landscape.space.index_of_freq(330.0).unwrap();
    for idx in target - 1..=target + 1 {
        landscape.consonance_field_score[idx] = 1.0;
        landscape.consonance_field_level[idx] = 1.0;
        landscape.consonance_field_score_eff[idx] = 1.0;
        landscape.consonance_field_level_eff[idx] = 1.0;
    }
    for (chain, moves) in [(".seek_consonance()", true), ("", false)] {
        let (scenario, _) =
            run_script_with_seed(&format!("place(sine().sustain(){chain},at(220.0));"), 21);
        let mut community = Community::new(Timebase {
            fs: 48_000.0,
            hop: 64,
        });
        community.set_seed(scenario.seed);
        for action in scenario.events.iter().flat_map(|event| &event.actions) {
            if matches!(action, Action::Spawn { .. }) {
                community.apply_action(action.clone(), &landscape);
            }
        }
        let voice = &mut community.voices[0];
        assert_eq!(voice.body.base_freq_hz(), 220.0);
        voice.set_accumulated_time_for_test(5.0);
        voice.set_theta_phase_state_for_test(6.0, true);
        let mut rhythms = NeuralRhythms::default();
        rhythms.theta.mag = 1.0;
        rhythms.theta.phase = 0.25;
        let initial = voice.target_pitch_log2();
        voice.update_pitch_target(&rhythms, 0.01, &landscape, &[], &[]);
        if moves {
            assert!(voice.target_pitch_log2() > initial);
        } else {
            assert_eq!(voice.target_pitch_log2(), initial);
        }
    }
}

#[test]
fn at_seek_does_not_change_create_or_live_freq_lock() {
    let (scenario, _) = run_script(
        "let spec = sine().freq(110.0).seek_consonance();\
         create(spec,1); let p = place(spec,at(330.0)); p.freq(440.0); flush();",
    );
    let mut community = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario.events.iter().flat_map(|event| &event.actions) {
        community.apply_action(action.clone(), &landscape);
    }
    assert_eq!(community.voices.len(), 2);
    assert_eq!(
        community.voices[0].effective_control.pitch.mode,
        PitchMode::Free
    );
    assert_eq!(community.voices[0].effective_control.pitch.freq, 110.0);
    assert_eq!(
        community.voices[1].effective_control.pitch.mode,
        PitchMode::Lock
    );
    assert_eq!(community.voices[1].effective_control.pitch.freq, 440.0);
}

#[test]
fn at_seek_keeps_supported_tie_and_persistence_without_ignored_warnings() {
    for timing in ["entrained()", "steady()", "flow()"] {
        for mode in ["gate_snap", "rearticulate"] {
            let persistence = if mode == "rearticulate" {
                ".rearticulate_above(75).articulation_persistence(0.75)"
            } else {
                ""
            };
            let (scenario, warnings) = run_script(&format!(
                "place(harmonic().sustain().{timing}.seek_consonance().pitch_apply_mode(\"{mode}\").tie(0.5){persistence},at(330.0));"
            ));
            assert!(warnings.lock().unwrap().is_empty());
            let spec = scenario
                .events
                .iter()
                .flat_map(|event| &event.actions)
                .find_map(|action| match action {
                    Action::Spawn { spec, .. } => Some(spec),
                    _ => None,
                })
                .unwrap();
            assert!(spec.control.supports_tie());
            assert_eq!(spec.control.phonation.tie, 0.5);
            if mode == "rearticulate" {
                assert!(spec.control.supports_articulation_persistence());
                assert_eq!(spec.control.phonation.articulation_persistence, 0.75);
            }
        }
    }
    let (_, warnings) = run_script(
        "place(harmonic().sustain().entrained().anchor().pitch_apply_mode(\"rearticulate\").rearticulate_above(75).articulation_persistence(0.75).tie(0.5),at(330.0));",
    );
    let warnings = warnings.lock().unwrap();
    assert!(
        warnings
            .iter()
            .any(|warning| warning.message.contains("tie() ignored"))
    );
    assert!(warnings.iter().any(|warning| {
        warning
            .message
            .contains("articulation_persistence() ignored")
    }));
    let (scenario, warnings) = run_script(
        "place(harmonic().seek_consonance().respawn_random().respawn_settle(at(440.0)),at(330.0));",
    );
    assert!(
        warnings
            .lock()
            .unwrap()
            .iter()
            .any(|warning| warning.message.contains("respawn_settle() requires"))
    );
    let settle = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::SetRespawnPolicy {
                settle_strategy, ..
            } => Some(settle_strategy),
            _ => None,
        })
        .unwrap();
    assert!(settle.is_none());
}

#[test]
fn consonance_movement_sets_free_hill_climb_glide_defaults() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().seek_consonance(), 1);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned voice");
    assert_eq!(voice.effective_control.pitch.mode, PitchMode::Free);
    assert_eq!(
        voice.effective_control.pitch.core_kind,
        PitchCoreKind::HillClimb
    );
    assert_eq!(
        voice.effective_control.pitch.pitch_apply_mode,
        PitchApplyMode::Glide
    );
    assert!((voice.effective_control.pitch.pitch_glide_tau_sec - 0.30).abs() <= 1e-6);
}

#[test]
fn consonance_movement_snaps_on_reattacking_phonation() {
    for preset in ["metric()", "entrained()", "flow()", "repeat().pulse(2.0)"] {
        let spec = first_spawn_spec_for_script(&format!(
            r#"
                create(sine().{preset}.seek_consonance(), 1);
                flush();
            "#
        ));
        assert_eq!(
            spec.control.pitch.pitch_apply_mode,
            PitchApplyMode::GateSnap,
            "seek_consonance with {preset} should snap at onsets"
        );
    }
}

#[test]
fn movement_glide_overrides_consonance_movement_default() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().seek_consonance().glide(0.12), 1);
            flush();
        "#,
    );
    let control = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.control.clone()),
            _ => None,
        })
        .expect("spawn action");
    assert_eq!(control.pitch.pitch_apply_mode, PitchApplyMode::Glide);
    assert!((control.pitch.pitch_glide_tau_sec - 0.12).abs() <= 1e-6);
}

#[test]
fn population_spec_pitch_core_sets_spawn_core_kind() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().pitch_core("peak_sampler"), 1);
            flush();
        "#,
    );
    let core_kind = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.control.pitch.core_kind),
            _ => None,
        })
        .expect("spawn action");
    assert_eq!(core_kind, PitchCoreKind::PeakSampler);
}

#[test]
fn test_create_accepts_preconfigured_population_spec() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().landscape_weight(0.25), 1);
            flush();
        "#,
    );
    let weight = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.control.pitch.landscape_weight),
            _ => None,
        })
        .expect("spawn action");
    assert!((weight - 0.25).abs() <= 1e-6);
}

#[test]
fn population_spec_landscape_weight_reaches_spawned_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().landscape_weight(0.3), 1);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.effective_control.pitch.landscape_weight - 0.3).abs() <= 1e-6);
}

#[test]
fn population_spec_temperature_reaches_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().temperature(0.3), 1);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().temperature_for_test() - 0.3).abs() <= 1e-6);
}

#[test]
fn population_landscape_weight_emits_live_update() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.landscape_weight(0.4);
            flush();
        "#,
    );
    let mut saw_update = false;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::UpdatePopulation { patch, .. } = action
            && patch.landscape_weight == Some(0.4)
        {
            saw_update = true;
            break;
        }
    }
    assert!(saw_update, "expected landscape_weight live update");
}

#[test]
fn population_landscape_weight_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.landscape_weight(0.6);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdatePopulation { .. } => {
                pop.apply_action(action.clone(), &landscape);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.effective_control.pitch.landscape_weight - 0.6).abs() <= 1e-6);
}

#[test]
fn population_amp_live_update_preserves_member_pitch_centers() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = place(sine(), consonance(196.0).peak().range(0.8, 2.5).spacing(0.9).count(4));
            flush();
            let g = g.amp(0.33);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    let mut before_update: Vec<(u64, f32)> = Vec::new();
    let mut after_update: Vec<(u64, f32)> = Vec::new();

    for event in &scenario.events {
        for action in &event.actions {
            pop.apply_action(action.clone(), &landscape);
        }
        if event
            .actions
            .iter()
            .any(|action| matches!(action, Action::Spawn { .. }))
        {
            for (idx, voice) in pop.voices.iter_mut().enumerate() {
                let freq_hz = 220.0 * (idx as f32 + 1.0);
                voice.force_set_pitch_log2(freq_hz.log2());
            }
            before_update = pop
                .voices
                .iter()
                .map(|voice| (voice.id(), voice.body.base_freq_hz()))
                .collect();
            before_update.sort_by_key(|(id, _)| *id);
        }
        if event
            .actions
            .iter()
            .any(|action| matches!(action, Action::UpdatePopulation { .. }))
        {
            after_update = pop
                .voices
                .iter()
                .map(|voice| (voice.id(), voice.body.base_freq_hz()))
                .collect();
            after_update.sort_by_key(|(id, _)| *id);
        }
    }

    assert_eq!(before_update.len(), 4);
    assert_eq!(before_update.len(), after_update.len());

    for ((id_a, freq_a), (id_b, freq_b)) in before_update.iter().zip(after_update.iter()) {
        assert_eq!(*id_a, *id_b);
        assert!(
            (freq_a - freq_b).abs() <= 1e-6,
            "amp-only live update must not modify member pitch center"
        );
    }
}

#[test]
fn population_temperature_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.temperature(0.4);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdatePopulation { .. } => {
                pop.apply_action(action.clone(), &landscape);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().temperature_for_test() - 0.4).abs() <= 1e-6);
}

#[test]
fn population_spec_crowding_sets_spawn_control() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().avoid_neighbors(1.2, 35.0), 1);
            flush();
        "#,
    );
    let (strength, sigma_cents) = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some((
                spec.control.pitch.crowding_strength,
                spec.control.pitch.crowding_sigma_cents,
            )),
            _ => None,
        })
        .expect("spawn action");
    assert!((strength - 1.2).abs() <= 1e-6);
    assert!((sigma_cents - 35.0).abs() <= 1e-6);
}

#[test]
fn population_spec_crowding_reaches_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().avoid_neighbors(1.2, 35.0), 1);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().crowding_strength_for_test() - 1.2).abs() <= 1e-6);
    assert!((voice.pitch_core_for_test().crowding_sigma_cents_for_test() - 35.0).abs() <= 1e-3);
}

#[test]
fn population_spec_crowding_single_arg_uses_default_sigma() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().avoid_neighbors(0.8), 1);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().crowding_strength_for_test() - 0.8).abs() <= 1e-6);
    assert!((voice.pitch_core_for_test().crowding_sigma_cents_for_test() - 60.0).abs() <= 1e-3);
}

#[test]
fn population_spec_crowding_mixed_numeric_overloads_work() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().avoid_neighbors(1, 35.0), 1);
            create(sine().avoid_neighbors(1.0, 35), 1);
            flush();
        "#,
    );
    let mut seen = 0usize;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { spec, .. } = action {
            assert!((spec.control.pitch.crowding_strength - 1.0).abs() <= 1e-6);
            assert!((spec.control.pitch.crowding_sigma_cents - 35.0).abs() <= 1e-6);
            seen += 1;
        }
    }
    assert_eq!(seen, 2);
}

#[test]
fn population_crowding_live_update_reaches_individual_core() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.avoid_neighbors(0.8, 25.0);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdatePopulation { .. } => {
                pop.apply_action(action.clone(), &landscape);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().crowding_strength_for_test() - 0.8).abs() <= 1e-6);
    assert!((voice.pitch_core_for_test().crowding_sigma_cents_for_test() - 25.0).abs() <= 1e-3);
}

#[test]
fn population_crowding_mixed_numeric_overloads_work() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.avoid_neighbors(1, 35.0);
            flush();
            let g = g.avoid_neighbors(1.0, 35);
            flush();
        "#,
    );
    let mut updates = 0usize;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::UpdatePopulation { patch, .. } = action
            && let (Some(strength), Some(sigma)) =
                (patch.crowding_strength, patch.crowding_sigma_cents)
        {
            assert!((strength - 1.0).abs() <= 1e-6);
            assert!((sigma - 35.0).abs() <= 1e-6);
            updates += 1;
        }
    }
    assert_eq!(updates, 2);
}

#[test]
fn population_crowding_target_emits_actions_for_founders_and_live_updates() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine().crowding_target(true, false), 1);
            flush();
            let g = g.crowding_target(true, true);
            flush();
        "#,
    );
    let mut saw_spawn_target = false;
    let mut saw_live_target = false;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::SetPopulationCrowdingTarget {
            same_population_visible,
            other_population_visible,
            ..
        } = action
        {
            if *same_population_visible && !*other_population_visible {
                saw_spawn_target = true;
            }
            if *same_population_visible && *other_population_visible {
                saw_live_target = true;
            }
        }
    }
    assert!(
        saw_spawn_target,
        "expected founder crowding_target to be emitted"
    );
    assert!(
        saw_live_target,
        "expected live crowding_target update to be emitted"
    );
}

#[test]
fn population_spec_leave_self_out_and_temperature_reach_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().leave_self_out(true).temperature(0.12), 1);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!(voice.pitch_core_for_test().leave_self_out_for_test());
    assert!((voice.pitch_core_for_test().temperature_for_test() - 0.12).abs() <= 1e-6);
}

#[test]
fn population_leave_self_out_and_temperature_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.leave_self_out(true).temperature(0.2);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdatePopulation { .. } => {
                pop.apply_action(action.clone(), &landscape);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!(voice.pitch_core_for_test().leave_self_out_for_test());
    assert!((voice.pitch_core_for_test().temperature_for_test() - 0.2).abs() <= 1e-6);
}

#[test]
fn population_spec_move_cost_reaches_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().move_cost(0.9), 1);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().move_cost_coeff_for_test() - 0.9).abs() <= 1e-6);
}

#[test]
fn population_move_cost_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.move_cost(0.8);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdatePopulation { .. } => {
                pop.apply_action(action.clone(), &landscape);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().move_cost_coeff_for_test() - 0.8).abs() <= 1e-6);
}

#[test]
fn population_hill_climb_knobs_and_loo_live_update_reach_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g
                .neighbor_step_cents(25)
                .tessitura_gravity(0.12)
                .move_cost_exp(2)
                .leave_self_out(true)
                .leave_self_out(true);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdatePopulation { .. } => {
                pop.apply_action(action.clone(), &landscape);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    let core = voice.pitch_core_for_test();
    assert!((core.neighbor_step_cents_for_test() - 25.0).abs() <= 1e-6);
    assert!((core.tessitura_gravity_for_test() - 0.12).abs() <= 1e-6);
    assert_eq!(core.move_cost_exp_for_test(), 2);
    assert!(core.leave_self_out_for_test());
    assert!(core.leave_self_out_for_test());
    assert!(voice.effective_control.pitch.leave_self_out);
}

#[test]
fn population_spec_peak_sampler_knobs_reach_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                sine()                    .pitch_core("peak_sampler")
                    .neighbor_step_cents(30)
                    .tessitura_gravity(0.14)
                    .window_cents(320)
                    .top_k(7)
                    .temperature(0.0)
                    .sigma_cents(18)
                    .random_candidates(5),
                1
            );
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    let core = voice.pitch_core_for_test();
    assert!((core.neighbor_step_cents_for_test() - 30.0).abs() <= 1e-6);
    assert!((core.tessitura_gravity_for_test() - 0.14).abs() <= 1e-6);
    assert!((core.window_cents_for_test() - 320.0).abs() <= 1e-6);
    assert_eq!(core.top_k_for_test(), 7);
    assert!((core.temperature_for_test() - 0.0).abs() <= 1e-6);
    assert!((core.sigma_cents_for_test() - 18.0).abs() <= 1e-6);
    assert_eq!(core.random_candidates_for_test(), 5);
}

#[test]
fn population_spec_advanced_pitch_knobs_reach_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                sine()                    .proposal_interval(0.3)
                    .global_peaks(12, 40.0)
                    .ratio_candidates(5)
                    .move_cost_time_scale("proposal_interval")
                    .pitch_apply_mode("glide")
                    .glide(0.08),
                1
            );
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    let core = voice.pitch_core_for_test();
    assert!((core.proposal_interval_sec_for_test().unwrap_or(0.0) - 0.3).abs() <= 1e-6);
    assert_eq!(core.global_peak_count_for_test(), 12);
    assert_eq!(core.ratio_candidate_count_for_test(), 5);
    assert!(core.use_ratio_candidates_for_test());
    assert_eq!(
        core.move_cost_time_scale_for_test(),
        MoveCostTimeScale::ProposalInterval
    );
    assert_eq!(
        voice.effective_control.pitch.pitch_apply_mode,
        PitchApplyMode::Glide
    );
    assert!((voice.effective_control.pitch.pitch_glide_tau_sec - 0.08).abs() <= 1e-6);
}

#[test]
fn population_advanced_pitch_knobs_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g
                .proposal_interval(0.25)
                .global_peaks(10, 30)
                .ratio_candidates(4)
                .move_cost_time_scale("proposal_interval")
                .pitch_apply_mode("glide")
                .glide(0.05);
            flush();
        "#,
    );
    let mut pop = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = Landscape::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdatePopulation { .. } => {
                pop.apply_action(action.clone(), &landscape);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    let core = voice.pitch_core_for_test();
    assert!((core.proposal_interval_sec_for_test().unwrap_or(0.0) - 0.25).abs() <= 1e-6);
    assert_eq!(core.global_peak_count_for_test(), 10);
    assert_eq!(core.ratio_candidate_count_for_test(), 4);
    assert!(core.use_ratio_candidates_for_test());
    assert_eq!(
        core.move_cost_time_scale_for_test(),
        MoveCostTimeScale::ProposalInterval
    );
    assert_eq!(
        voice.effective_control.pitch.pitch_apply_mode,
        PitchApplyMode::Glide
    );
    assert!((voice.effective_control.pitch.pitch_glide_tau_sec - 0.05).abs() <= 1e-6);
}

#[test]
fn set_pitch_objective_emits_landscape_update() {
    let (scenario, _warnings) = run_script(
        r#"
            set_pitch_objective("dissonance");
            wait(0.1);
            set_pitch_objective("consonance");
        "#,
    );
    let modes: Vec<PitchObjectiveMode> = scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
        .filter_map(|action| match action {
            Action::UpdateLandscape { update } => update.pitch_objective_mode,
            _ => None,
        })
        .collect();
    assert_eq!(
        modes,
        vec![
            PitchObjectiveMode::NegativeConsonance,
            PitchObjectiveMode::Consonance
        ]
    );
}

#[test]
fn population_spec_landscape_weight_sets_spawn_control() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().landscape_weight(0.4), 1);
        "#,
    );
    let weight = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.control.pitch.landscape_weight),
            _ => None,
        })
        .expect("spawn action");
    assert!((weight - 0.4).abs() <= 1e-6);
}

#[test]
fn population_spec_respawn_policy_emits_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().respawn_hereditary(0.03), 1);
            flush();
        "#,
    );
    let mut saw_policy = false;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::SetRespawnPolicy {
            population_id,
            policy,
            ..
        } = action
        {
            assert_eq!(*population_id, 1);
            assert_eq!(*policy, RespawnPolicy::Hereditary { sigma_oct: 0.03 });
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy action");
}

#[test]
fn population_spec_respawn_random_emits_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().respawn_random(), 1);
        "#,
    );
    let mut saw_policy = false;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::SetRespawnPolicy {
            population_id,
            policy,
            ..
        } = action
        {
            assert_eq!(*population_id, 1);
            assert_eq!(*policy, RespawnPolicy::Random);
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy(Random)");
}

#[test]
fn respawn_capacity_defaults_to_founder_count() {
    let (scenario, _warnings) = run_script(
        r#"
            place(sine().respawn_random(), line(220.0, 440.0).count(3));
        "#,
    );
    let capacity = scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
        .find_map(|action| match action {
            Action::SetRespawnPolicy { capacity, .. } => Some(*capacity),
            _ => None,
        })
        .expect("respawn policy action");
    assert_eq!(capacity, 3);
}

#[test]
fn population_spec_respawn_hereditary_emits_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().respawn_hereditary(0.03), 1);
        "#,
    );
    let mut saw_policy = false;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::SetRespawnPolicy {
            population_id,
            policy,
            ..
        } = action
        {
            assert_eq!(*population_id, 1);
            assert_eq!(*policy, RespawnPolicy::Hereditary { sigma_oct: 0.03 });
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy(Hereditary)");
}

#[test]
fn population_spec_respawn_consonance_emits_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().respawn_consonance(), 1);
        "#,
    );
    let mut saw_policy = false;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::SetRespawnPolicy {
            population_id,
            policy: RespawnPolicy::PeakBiased { config },
            ..
        } = action
        {
            assert_eq!(*population_id, 1);
            assert_eq!(*config, RespawnPeakBiasConfig::default());
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy(PeakBiased)");
}

#[test]
fn population_respawn_tier2_settings_reach_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                sine()
                    .respawn_hereditary(0.03)
                    .respawn_settle(consonance(220.0).peak().range(0.75, 1.5).spacing(0.5))
                    .respawn_capacity(3)
                    .respawn_min_consonance(0.4)
                    .respawn_background_death_rate(0.03),
                1
            );
        "#,
    );
    let mut saw_policy = false;
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::SetRespawnPolicy {
            population_id,
            policy,
            settle_strategy,
            capacity,
            min_c_level,
            background_death_rate_per_sec,
        } = action
        {
            assert_eq!(*population_id, 1);
            assert_eq!(*policy, RespawnPolicy::Hereditary { sigma_oct: 0.03 });
            assert_eq!(*capacity, 3);
            assert_eq!(*min_c_level, Some(0.4));
            assert!((*background_death_rate_per_sec - 0.03).abs() <= 1e-6);
            assert!(matches!(
                settle_strategy,
                Some(SpawnStrategy::Field {
                    target: FieldTarget::Consonance,
                    sampling: FieldSampling::Peak,
                    min_freq,
                    max_freq,
                    min_dist_erb,
                    ..
                }) if (*min_freq - 220.0 * 0.75).abs() <= 1e-4
                    && (*max_freq - 220.0 * 1.5).abs() <= 1e-4
                    && (*min_dist_erb - 0.5).abs() <= 1e-6
            ));
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy with tier-2 settings");
}

#[test]
fn spawn_payload_preserves_population_spec_control_fields() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                harmonic()
                    .amp(0.33)
                    .freq(330.0)
                    .brightness(0.7),
                1
            );
        "#,
    );
    let control = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, strategy, .. } => {
                assert!(strategy.is_none());
                Some(spec.control.clone())
            }
            _ => None,
        })
        .expect("spawn action");
    assert!((control.body.amp - 0.33).abs() <= 1e-6);
    assert!((control.pitch.freq - 330.0).abs() <= 1e-6);
    assert_eq!(control.pitch.mode, PitchMode::Lock);
    assert!((control.body.timbre.brightness - 0.7).abs() <= 1e-6);
}

#[test]
fn spawn_payload_preserves_consonance_viability_window() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                harmonic()                    .endurance(10.0)
                    .recovery(3.3333333)
                    .consonance_viability(0.3, 0.8),
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
        continuous_recharge_score_low,
        continuous_recharge_score_high,
        selection_environment,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert_eq!(continuous_recharge_score_low, Some(0.3));
    assert_eq!(continuous_recharge_score_high, Some(0.8));
    assert!(selection_environment);
}

#[test]
fn lifecycle_seconds_resolve_to_normalized_runtime_rates() {
    let spawn = first_spawn_spec_for_script(
        r#"
            create(
                harmonic()
                    .endurance(8.0)
                    .recovery(4.0)
                    .attack_cost_fraction(0.05)
                    .attack_recharge_fraction(0.20)
                    .dissonance_penalty(3.0),
                1
            );
            flush();
        "#,
    );
    let ArticulationCoreConfig::Entrain { lifecycle, .. } = &spawn.articulation else {
        panic!("expected entrain articulation");
    };
    let LifecycleConfig::Sustain {
        endurance_sec,
        recovery_sec,
        attack_cost_fraction,
        attack_recharge_fraction,
        dissonance_penalty,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert_eq!(*endurance_sec, Some(8.0));
    assert_eq!(*recovery_sec, Some(4.0));
    assert_eq!(*attack_cost_fraction, Some(0.05));
    assert_eq!(*attack_recharge_fraction, Some(0.20));
    assert_eq!(*dissonance_penalty, 3.0);

    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    let core = AnyArticulationCore::from_config(&spawn.articulation, 48_000.0, 9, &mut rng);
    let AnyArticulationCore::Entrain(core) = core else {
        panic!("expected entrain core");
    };
    assert_eq!(core.endurance_sec, Some(8.0));
    assert_eq!(core.recovery_sec, Some(4.0));
    assert!((core.metabolism.basal_cost_per_sec - 1.0 / 32.0).abs() < 1e-6);
    assert!((core.metabolism.continuous_recharge_per_sec - 0.25).abs() < 1e-6);
    assert!((core.metabolism.action_cost_per_attack - 0.05).abs() < 1e-6);
    assert!((core.metabolism.recharge_per_attack - 0.20).abs() < 1e-6);
}

#[test]
fn spawn_payload_preserves_selection_environment() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                harmonic()                    .endurance(10.0)
                    .viability_scope("environment"),
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
    assert!(selection_environment);
}

#[test]
fn selection_environment_can_override_consonance_viability_for_reference_assays() {
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
fn timbre_method_is_not_registered() {
    let err = run_script_err(
        r#"
            create(harmonic(), 1).timbre(0.7, 0.2);
        "#,
    );
    assert!(err.message.contains("timbre"));
}

#[test]
fn width_method_is_not_registered() {
    let err = run_script_err(
        r#"
            create(harmonic(), 1).width(0.4);
        "#,
    );
    assert!(err.message.contains("width"));
}

#[test]
fn removed_pre_040_api_names_are_not_registered() {
    for (script, name) in [
        (r#"harmonic().crowding(0.5);"#, "crowding"),
        (r#"harmonic().movement_glide(0.2);"#, "movement_glide"),
        (r#"harmonic().pitch_glide(0.2);"#, "pitch_glide"),
        (r#"set_harmonic_mirror(0.5);"#, "set_harmonic_mirror"),
        (r#"derive(harmonic());"#, "derive"),
        (r#"consonance_density(80.0, 900.0);"#, "consonance_density"),
        (r#"harmonic("name");"#, "harmonic"),
        (r#"variant("name", harmonic());"#, "variant"),
        (r#"harmonic().send(field);"#, "field"),
        (r#"harmonic().send(field_bus);"#, "field_bus"),
        (
            r#"harmonic().send(generator_field_bus);"#,
            "generator_field_bus",
        ),
        (r#"harmonic().send(presentation);"#, "presentation"),
        (r#"harmonic().accent(0.5);"#, "accent"),
        (r#"harmonic().sync(0.5);"#, "sync"),
        (r#"harmonic().gates(2);"#, "gates"),
        (r#"harmonic().field();"#, "field"),
        (r#"harmonic().field_window(0.2, 0.8);"#, "field_window"),
        (r#"harmonic().field_curve(2.0, 0.5);"#, "field_curve"),
        (r#"harmonic().field_drop(0.5);"#, "field_drop"),
        (r#"harmonic().metabolism(0.1);"#, "metabolism"),
        (r#"harmonic().initial_energy(1.0);"#, "initial_energy"),
        (r#"harmonic().energy_cap(1.0);"#, "energy_cap"),
        (r#"harmonic().recharge_rate(0.1);"#, "recharge_rate"),
        (r#"harmonic().action_cost(0.1);"#, "action_cost"),
        (r#"harmonic().viability_rate(0.1);"#, "viability_rate"),
        (r#"harmonic().dissonance_cost(0.1);"#, "dissonance_cost"),
    ] {
        let err = run_script_err(script);
        assert!(
            err.message.contains(name),
            "expected error for removed API name {name}, got {}",
            err.message
        );
    }
}

#[test]
fn unknown_brain_value_fails_fast_on_population_spec() {
    let err = run_script_err(r#"harmonic().brain("nonexistent");"#);
    assert!(
        err.message.contains("brain") && err.message.contains("nonexistent"),
        "expected an unknown-brain error, got {}",
        err.message
    );
}

#[test]
fn unknown_rhythm_role_value_fails_fast_on_population_spec() {
    let err = run_script_err(r#"harmonic().rhythm_role("nonexistent");"#);
    assert!(
        err.message.contains("rhythm_role") && err.message.contains("nonexistent"),
        "expected an unknown-rhythm_role error, got {}",
        err.message
    );
}

#[test]
fn spread_and_unison_methods_update_timbre() {
    let (scenario, _warnings) = run_script(
        r#"
            create(harmonic().spread(0.4).unison(5), 1);
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

    assert!((spawn.control.body.timbre.spread - 0.4).abs() <= 1.0e-6);
    assert_eq!(spawn.control.body.timbre.unison, 5);
}

#[test]
fn rhythm_modulators_are_sanitized_at_core_boundary() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                sine()                    .rhythm_coupling_vitality(-3.0, 2.0)
                    .rhythm_reward(-2.0, "attack_phase_match"),
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
    let mut rng = rand::rngs::StdRng::seed_from_u64(9);
    let core = AnyArticulationCore::from_config(&spawn.articulation, 48_000.0, 9, &mut rng);
    let AnyArticulationCore::Entrain(core) = core else {
        panic!("expected entrain core");
    };

    assert_eq!(
        core.rhythm_coupling,
        RhythmCouplingMode::TemporalTimesVitality {
            lambda_v: 0.0,
            v_floor: 0.999
        }
    );
    let reward = core.rhythm_reward.expect("expected rhythm reward");
    assert_eq!(reward.rho_t, 0.0);
}

#[test]
fn metric_founders_read_the_final_scene_basin_through_create_and_place() {
    for placement in [
        "create(sine().metric(), 2);",
        "place(sine().metric(), line(220.0, 440.0).count(2));",
    ] {
        for script in [
            format!("temporal_basin(2.8, 3.4); {placement}"),
            format!("{placement} wait(1.0); temporal_basin(2.8, 3.4);"),
            format!(
                "temporal_basin(1.2, 1.6); {placement} wait(1.0); temporal_basin(2.8, 3.4); {placement}"
            ),
        ] {
            let (scenario, _) = run_script(&script);
            assert_eq!(scenario.meter_shaping.stability, 0.0);
            assert_eq!(scenario.meter_shaping.basin_hz, Some((2.8, 3.4)));
            let mut founders = 0;
            for action in scenario.events.iter().flat_map(|event| &event.actions) {
                if let Action::Spawn { spec, ids, .. } = action {
                    let PhonationTiming::Coupled(timing) = &spec.control.phonation.spec.timing
                    else {
                        panic!("expected metric timing");
                    };
                    assert_eq!(timing.base_rate_hz, 0.5 * (2.8 + 3.4));
                    founders += ids.len();
                }
            }
            assert!(founders >= 2);
        }
    }
    let spawn = first_spawn_spec_for_script("create(sine().metric(), 1);");
    let PhonationTiming::Coupled(timing) = spawn.control.phonation.spec.timing else {
        panic!("expected metric timing");
    };
    assert_eq!(timing.base_rate_hz, 2.0);
    assert_eq!(CoupledTimingSpec::default().base_rate_hz, 2.0);
}

#[test]
fn metric_basin_sanitization_matches_the_listener_center() {
    for (a, b) in [(3.4, 2.8), (-1.0, 8.0), (0.5, 0.5), (4.0, 4.0)] {
        let spawn = first_spawn_spec_for_script(&format!(
            "create(sine().metric(), 1); temporal_basin({a}, {b});"
        ));
        let PhonationTiming::Coupled(timing) = spawn.control.phonation.spec.timing else {
            panic!("expected metric timing");
        };
        let (lo, hi) = sane_basin(a, b);
        assert_eq!(timing.base_rate_hz, 0.5 * (lo + hi));
    }
}

#[test]
fn basin_preserves_participating_and_explicit_pulse_rates() {
    for preset in ["entrained", "flow"] {
        let spawn = first_spawn_spec_for_script(&format!(
            "create(sine().{preset}(), 1); temporal_basin(2.8, 3.4);"
        ));
        let PhonationTiming::Coupled(timing) = spawn.control.phonation.spec.timing else {
            panic!("expected participating timing");
        };
        assert_eq!(timing.relation, RhythmRelation::Participating);
        assert_eq!(timing.base_rate_hz, 2.0);
    }
    let spawn = first_spawn_spec_for_script(
        "create(sine().repeat().pulse(1.7), 1); temporal_basin(2.8, 3.4);",
    );
    assert!(matches!(spawn.control.phonation.spec.timing,
        PhonationTiming::Pulse { rate_hz, .. } if rate_hz == 1.7));
}

#[test]
fn metric_respawn_starts_from_the_resolved_population_template() {
    let (scenario, _) = run_script_with_seed(
        "create(sine().metric().anchor().respawn_random().respawn_background_death_rate(10000.0), 1); \
         temporal_basin(2.8, 3.4);",
        21,
    );
    let mut population = Community::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    population.set_seed(scenario.seed);
    let landscape = Landscape::default();
    for action in scenario.events.iter().flat_map(|event| &event.actions) {
        if matches!(
            action,
            Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
        ) {
            population.apply_action(action.clone(), &landscape);
        }
    }
    let founder = population.voices[0].id();
    let period = population.voices[0]
        .phonation_engine
        .clock
        .intrinsic_period_sec();
    assert_eq!(period, Some(1.0 / f64::from(0.5 * (2.8f32 + 3.4))));
    // A child's template rate must not become the parent's current clock rate.
    let mut changed = population.voices[0]
        .effective_control
        .phonation
        .spec
        .clone();
    let PhonationTiming::Coupled(timing) = &mut changed.timing else {
        panic!("expected metric timing");
    };
    timing.base_rate_hz = 1.3;
    population.voices[0].phonation_engine =
        crate::life::phonation_engine::PhonationEngine::from_spec(&changed, 21);
    assert_ne!(
        population.voices[0]
            .phonation_engine
            .clock
            .intrinsic_period_sec(),
        period
    );
    population.advance(64, 0, 64.0 / 48_000.0, &landscape);
    population.cleanup_dead(0, 64.0 / 48_000.0, false, &landscape);
    assert_eq!(population.voices.len(), 1);
    assert_ne!(population.voices[0].id(), founder);
    assert_eq!(
        population.voices[0]
            .phonation_engine
            .clock
            .intrinsic_period_sec(),
        period
    );
}

#[test]
fn entrained_sets_defaults_and_accepts_prior_social_tuning() {
    let spawn = first_spawn_spec_for_script(
        r#"
            create(sine().entrained().cycles(2), 1);
            flush();
        "#,
    );
    let PhonationTiming::Coupled(spec) = spawn.control.phonation.spec.timing else {
        panic!("expected coupled timing intent");
    };
    assert!(spec.coupling > 0.0 && spec.coupling < 1.0);
    assert_eq!(
        spec.relation,
        crate::scenario::RhythmRelation::Participating
    );
    assert!(spec.social > 0.0);
    let ArticulationCoreConfig::Entrain {
        rhythm_coupling,
        rhythm_reward,
        ..
    } = spawn.articulation
    else {
        panic!("expected entrain articulation");
    };
    assert!(matches!(
        rhythm_coupling,
        RhythmCouplingMode::TemporalTimesVitality { .. }
    ));
    assert!(rhythm_reward.is_some());

    let tuned = first_spawn_spec_for_script(
        r#"
            create(sine().social(0.25).entrained().cycles(2), 1);
            flush();
        "#,
    );
    let PhonationTiming::Coupled(tuned_spec) = tuned.control.phonation.spec.timing else {
        panic!("expected coupled timing intent");
    };
    assert!((tuned_spec.social - 0.25).abs() <= 1e-6);
}

#[test]
fn while_alive_and_beat_modes_are_last_write_wins() {
    let hold = first_spawn_spec_for_script(
        r#"
            create(sine().metric().while_alive(), 1);
            flush();
        "#,
    );
    assert!(matches!(
        hold.control.phonation.spec.timing,
        PhonationTiming::Once
    ));
    assert!(matches!(
        hold.control.phonation.spec.duration,
        DurationSpec::WhileAlive
    ));
    let ArticulationCoreConfig::Entrain {
        rhythm_coupling,
        rhythm_reward,
        ..
    } = hold.articulation
    else {
        panic!("expected entrain articulation");
    };
    assert!(matches!(rhythm_coupling, RhythmCouplingMode::TemporalOnly));
    assert!(rhythm_reward.is_none());

    let beat = first_spawn_spec_for_script(
        r#"
            create(sine().while_alive().metric(), 1);
            flush();
        "#,
    );
    assert!(matches!(
        beat.control.phonation.spec.timing,
        PhonationTiming::Coupled(_)
    ));
    assert!(matches!(
        beat.control.phonation.spec.duration,
        DurationSpec::Gates(DEFAULT_GATE_COUNT)
    ));
}

#[test]
fn phonate_when_viable_sets_gate_on_population_spec() {
    let spawn = first_spawn_spec_for_script(
        r#"
            create(sine().phonate_when_viable(), 1);
        "#,
    );
    assert_eq!(spawn.control.phonation.gate, PhonationGate::WhenViable);
}

#[test]
fn initial_only_methods_are_not_registered_on_population() {
    for method in ["phonate_when_viable", "brain", "respawn_random"] {
        let script = format!("let population = place(sine(), at(220.0)); population.{method}();");
        let err = run_script_err(&script);
        assert!(
            err.message.contains(method) && err.message.contains("Population"),
            "expected {method} to be unavailable on Population, got {}",
            err.message
        );
    }
}

#[test]
fn flow_sets_low_coupling_intent() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().flow().cycles(1), 1);
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

    let PhonationTiming::Coupled(spec) = spawn.control.phonation.spec.timing else {
        panic!("expected coupled timing intent");
    };
    assert!(spec.coupling < 0.2, "flow should be near-zero coupling");
    assert_eq!(
        spec.relation,
        crate::scenario::RhythmRelation::Participating
    );
    assert!(spec.flow_depth > 0.0, "flow should jitter its renewal");
    assert_eq!(spec.role, RhythmRole::Texture);
}

#[test]
fn meter_stability_and_temporal_basin_set_scene_meter_shaping() {
    let (scenario, _warnings) = run_script(
        r#"
            meter_stability(0.7);
            temporal_basin(1.8, 2.4);
            create(sine().metric().cycles(1), 1);
            flush();
        "#,
    );
    assert!((scenario.meter_shaping.stability - 0.7).abs() <= 1e-6);
    let (min, max) = scenario.meter_shaping.basin_hz.expect("basin set");
    assert!((min - 1.8).abs() <= 1e-6 && (max - 2.4).abs() <= 1e-6);
}

#[test]
fn microtiming_sets_signed_beat_phase_offset() {
    let spawn = first_spawn_spec_for_script(
        r#"
            create(sine().metric().microtiming(-0.1).cycles(1), 1);
            flush();
        "#,
    );
    let PhonationTiming::Coupled(spec) = spawn.control.phonation.spec.timing else {
        panic!("expected coupled timing intent");
    };
    assert!((spec.microtiming + 0.1).abs() <= 1e-6);
}

#[test]
fn rearticulate_respawn_inherits_initial_and_live_population_template() {
    for live in [false, true] {
        let mode = if live { "glide" } else { "rearticulate" };
        let update = if live {
            "p.pitch_apply_mode(\"rearticulate\");"
        } else {
            ""
        };
        let (scenario, _) = run_script_with_seed(
            &format!(
                "let p = create(sine().sustain().seek_consonance().pitch_apply_mode(\"{mode}\").respawn_random().respawn_background_death_rate(10000.0), 1); {update} flush();"
            ),
            21,
        );
        let mut community = Community::new(Timebase {
            fs: 48_000.0,
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
        community.advance(257, 0, 257.0 / 48_000.0, &landscape);
        community.cleanup_dead(0, 257.0 / 48_000.0, false, &landscape);
        assert!(community.voices.iter().any(|v| v.id() != founder));
        for child in community.voices.iter().filter(|v| v.id() != founder) {
            assert_eq!(
                child.effective_control.pitch.pitch_apply_mode,
                PitchApplyMode::Rearticulate
            );
            assert!(child.reports_onset_pitch());
        }
    }
}

#[test]
fn synchronized_fixed_pitch_matches_sustained_participation_and_implicit_coupling_stays_unchanged()
{
    for preset in ["metric()", "entrained()", "flow()"] {
        for chain in [
            format!("anchor().sustain().{preset}"),
            format!("{preset}.sustain().anchor()"),
        ] {
            let spec = first_spawn_spec_for_script(&format!(
                "create(sine().{chain}.pitch_apply_mode(\"gate_snap\").tie(1),1);"
            ));
            assert!(spec.control.phonation.spec.is_sustained_participation());
            assert_eq!(
                spec.control.pitch.mode,
                crate::scenario::control::PitchMode::Lock
            );
            assert_eq!(spec.control.phonation.tie, 0.);
        }
    }
    for chain in [
        "entrainment(0.5).sustain()",
        "rhythm_role(\"beat\").sustain()",
        "metric().while_alive()",
        "while_alive().metric()",
        "sustain().metric().cycles(1)",
    ] {
        let spec = first_spawn_spec_for_script(&format!("create(sine().{chain},1);"));
        assert!(
            !spec.control.phonation.spec.is_sustained_participation(),
            "{chain}"
        );
    }
}

#[test]
fn synchronized_respawn_retains_controls_and_starts_fresh_private_policy() {
    use crate::life::shape_policy::Shape;
    let (scene, _) = run_script_with_seed(
        "create(sine().sustain().metric().pitch_apply_mode(\"rearticulate\").rearticulate_above(50).articulation_persistence(0.5).tie(1).respawn_random().respawn_background_death_rate(10000.0),1);",
        21,
    );
    let mut community = Community::new(Timebase {
        fs: 48000.,
        hop: 257,
    });
    community.set_seed(scene.seed);
    let landscape = Landscape::default();
    for action in scene.events.iter().flat_map(|e| &e.actions) {
        if matches!(
            action,
            Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
        ) {
            community.apply_action(action.clone(), &landscape);
        }
    }
    let parent = community.voices[0].id();
    let shape = community.voices[0]
        .phonation_engine
        .clock
        .shape_mut()
        .unwrap();
    shape.state = Shape::Connect;
    shape.has_tone = true;
    shape.tie.as_mut().unwrap().pending_pitch = true;
    community.advance(257, 0, 257. / 48000., &landscape);
    community.cleanup_dead(0, 257. / 48000., false, &landscape);
    assert!(community.voices.iter().any(|v| v.id() != parent));
    for child in community.voices.iter_mut().filter(|v| v.id() != parent) {
        assert!(
            child
                .effective_control
                .phonation
                .spec
                .is_sustained_participation()
        );
        assert_eq!(child.effective_control.phonation.tie, 1.);
        assert!(!child.phonation_engine.uses_participation());
        let shape = child.phonation_engine.clock.shape_mut().unwrap();
        assert_eq!(shape.state, Shape::Base);
        assert!(!shape.has_tone && !shape.veto_plan);
        assert!(!shape.tie.as_ref().unwrap().pending_pitch);
    }
}

#[test]
fn boundary_respawn_inherits_initial_and_live_population_template() {
    for live in [false, true] {
        let mode = if live { "glide" } else { "rearticulate" };
        let update = if live {
            "p.pitch_apply_mode(\"rearticulate\"); p.rearticulate_above(100);"
        } else {
            ""
        };
        let (scenario, _) = run_script_with_seed(
            &format!(
                "let p = create(sine().sustain().seek_consonance().pitch_apply_mode(\"{mode}\").rearticulate_above(100).respawn_random().respawn_background_death_rate(10000.0), 1); {update} flush();"
            ),
            21,
        );
        let mut community = Community::new(Timebase {
            fs: 48_000.0,
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
        community.advance(257, 0, 257.0 / 48_000.0, &landscape);
        community.cleanup_dead(0, 257.0 / 48_000.0, false, &landscape);
        assert!(community.voices.iter().any(|v| v.id() != founder));
        for child in community.voices.iter().filter(|v| v.id() != founder) {
            assert_eq!(
                child.effective_control.pitch.pitch_apply_mode,
                PitchApplyMode::Rearticulate
            );
            assert!(child.reports_onset_pitch());
            assert_eq!(
                child.effective_control.pitch.rearticulate_above_cents,
                100.0
            );
        }
    }
}

#[test]
fn deferred_participation_respawn_inherits_initial_and_live_population_template() {
    for live in [false, true] {
        let mode = if live { "glide" } else { "gate_snap" };
        let update = if live {
            "p.pitch_apply_mode(\"gate_snap\");"
        } else {
            ""
        };
        let (scenario, _) = run_script_with_seed(
            &format!(
                "let p = create(sine().sustain().entrained().seek_consonance().pitch_apply_mode(\"{mode}\").respawn_random().respawn_background_death_rate(10000.0), 1); {update} flush();"
            ),
            21,
        );
        let mut community = Community::new(Timebase {
            fs: 48_000.0,
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
        community.advance(257, 0, 257.0 / 48_000.0, &landscape);
        community.cleanup_dead(0, 257.0 / 48_000.0, false, &landscape);
        assert!(community.voices.iter().any(|v| v.id() != founder));
        for child in community.voices.iter().filter(|v| v.id() != founder) {
            assert_eq!(
                child.effective_control.pitch.pitch_apply_mode,
                PitchApplyMode::GateSnap
            );
            assert!(child.reports_onset_pitch());
            assert!(
                child
                    .effective_control
                    .phonation
                    .spec
                    .is_sustained_participation()
            );
            assert_eq!(child.effective_control.pitch.rearticulate_above_cents, 0.0);
        }
    }
}

#[test]
fn rest_numeric_overloads_and_sanitize_match_existing_vocabulary() {
    for (call, value) in [
        ("rest()", 1.),
        ("rest(0)", 0.),
        ("rest(0.5)", 0.5),
        ("rest(-2)", 0.),
        ("rest(2)", 2.),
    ] {
        let spec = first_spawn_spec_for_script(&format!(
            "create(sine().sustain().entrained().cycles(1).{call},1);"
        ));
        assert_eq!(spec.control.phonation.rest_periods, Some(value));
    }
    for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1., f32::MAX] {
        let mut p = PopulationSpec::preset(BodyMethod::Harmonic);
        p.set_rest(x);
        assert_eq!(
            p.rest_periods,
            Some(if x.is_finite() { x.max(0.) } else { 0. })
        );
    }
}

#[test]
fn rest_respawn_inherits_template_and_starts_with_empty_performance_state() {
    let (scenario, _) = run_script_with_seed(
        "create(sine().sustain().entrained().cycles(2).rest(0.5).pitch_apply_mode(\"gate_snap\").respawn_random().respawn_background_death_rate(10000.0),1);",
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
            Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
        ) {
            community.apply_action(action.clone(), &landscape);
        }
    }
    let parent = community.voices[0].id();
    community.voices[0]
        .phonation_engine
        .rest
        .as_mut()
        .unwrap()
        .phase = crate::life::phonation_engine::RestPhase::Gap {
        origin: 0,
        until: None,
    };
    community.advance(257, 0, 257. / 48000., &landscape);
    community.cleanup_dead(0, 257. / 48000., false, &landscape);
    let children: Vec<_> = community
        .voices
        .iter()
        .filter(|v| v.id() != parent)
        .collect();
    assert!(!children.is_empty());
    assert!(community.rest_enabled);
    for child in children {
        assert_eq!(child.effective_control.phonation.rest_periods, Some(0.5));
        assert_eq!(
            child.effective_control.phonation.finite_duration,
            Some(DurationSpec::Gates(2))
        );
        assert_eq!(
            child.phonation_engine.rest.unwrap().phase,
            crate::life::phonation_engine::RestPhase::Ready
        );
    }
}

#[test]
fn articulation_persistence_values_follow_existing_sanitize01() {
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
        let _ = engine.eval::<Dynamic>("create(sine().sustain().entrained().pitch_apply_mode(\"rearticulate\").rearticulate_above(50).articulation_persistence(test_value()),1);").unwrap();
        let state = ctx.lock().unwrap();
        assert_eq!(
            state
                .populations
                .values()
                .next()
                .unwrap()
                .spec
                .control
                .phonation
                .articulation_persistence,
            expected
        );
    }
    for value in ["0", "1", "0.0", "1.0"] {
        let spec = first_spawn_spec_for_script(&format!(
            "create(sine().sustain().flow().pitch_apply_mode(\"rearticulate\").rearticulate_above(50).articulation_persistence({value}),1);"
        ));
        assert_eq!(
            spec.control.phonation.articulation_persistence,
            value.parse::<f32>().unwrap()
        );
    }
}

#[test]
fn articulation_persistence_unsupported_spec_ignored_and_no_live_setter() {
    for chain in [
        "sustain()",
        "sustain().entrained()",
        "sustain().entrained().pitch_apply_mode(\"rearticulate\")",
        "sustain().entrained().anchor().pitch_apply_mode(\"rearticulate\").rearticulate_above(50)",
        "sustain().pitch_apply_mode(\"rearticulate\").rearticulate_above(50)",
        "pulse(2.0).cycles(1)",
    ] {
        let spec = first_spawn_spec_for_script(&format!(
            "create(sine().{chain}.articulation_persistence(0.75),1);"
        ));
        assert_eq!(
            spec.control.phonation.articulation_persistence, 0.0,
            "{chain}"
        );
    }
    let error = run_script_err(
        "let p=create(sine().sustain().entrained().pitch_apply_mode(\"rearticulate\").rearticulate_above(50).articulation_persistence(0.75),1); p.articulation_persistence(0.5);",
    );
    assert!(error.to_string().contains("articulation_persistence"));
}

#[test]
fn articulation_persistence_respawn_inherits_parameter_and_resets_private_policy() {
    use crate::life::phonation_engine::PhonationClock;
    use crate::life::shape_policy::Shape;
    let (scenario, _) = run_script_with_seed(
        "create(sine().sustain().entrained().pitch_apply_mode(\"rearticulate\").rearticulate_above(50).articulation_persistence(0.75).respawn_random().respawn_background_death_rate(10000.0),1);",
        21,
    );
    let mut community = Community::new(Timebase {
        fs: 48000.,
        hop: 257,
    });
    community.set_seed(scenario.seed);
    let landscape = Landscape::default();
    for action in scenario.events.iter().flat_map(|event| &event.actions) {
        if matches!(
            action,
            Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
        ) {
            community.apply_action(action.clone(), &landscape);
        }
    }
    let parent = community.voices[0].id();
    let PhonationClock::Participation(clock) = &mut community.voices[0].phonation_engine.clock
    else {
        panic!()
    };
    let shape = clock.shape.as_mut().unwrap();
    shape.state = Shape::Connect;
    shape.veto_plan = true;
    shape.has_tone = true;
    community.advance(257, 0, 257. / 48000., &landscape);
    community.cleanup_dead(0, 257. / 48000., false, &landscape);
    let children: Vec<_> = community
        .voices
        .iter()
        .filter(|v| v.id() != parent)
        .collect();
    assert!(!children.is_empty());
    for child in children {
        assert_eq!(
            child.effective_control.phonation.articulation_persistence,
            0.75
        );
        let PhonationClock::Participation(clock) = &child.phonation_engine.clock else {
            panic!()
        };
        let shape = clock.shape.as_ref().unwrap();
        assert_eq!(shape.state, Shape::Base);
        assert!(!shape.veto_plan && !shape.has_tone);
    }
}

#[test]
fn articulation_persistence_endpoints_allocate_no_private_policy() {
    use crate::life::phonation_engine::PhonationClock;
    for suffix in [
        "",
        ".articulation_persistence(0)",
        ".articulation_persistence(1)",
    ] {
        let (scenario, _) = run_script_with_seed(
            &format!(
                "create(sine().sustain().entrained().pitch_apply_mode(\"rearticulate\").rearticulate_above(50){suffix},1);"
            ),
            21,
        );
        let mut community = Community::new(Timebase {
            fs: 48000.,
            hop: 512,
        });
        for action in scenario.events.iter().flat_map(|event| &event.actions) {
            community.apply_action(action.clone(), &Landscape::default());
        }
        let PhonationClock::Participation(clock) = &community.voices[0].phonation_engine.clock
        else {
            panic!()
        };
        assert!(clock.shape.is_none());
    }
}

#[test]
fn synchronized_tie_default_resolves_omission_and_explicit_values_in_final_specs() {
    for body in ["sine", "harmonic", "modal"] {
        for mode in ["gate_snap", "rearticulate"] {
            let words = [
                "sustain()",
                "metric()",
                &format!("pitch_apply_mode(\"{mode}\")"),
            ];
            for order in [
                [0, 1, 2],
                [0, 2, 1],
                [1, 0, 2],
                [1, 2, 0],
                [2, 0, 1],
                [2, 1, 0],
            ] {
                let chain = order.map(|i| words[i]).join(".");
                for place in [false, true] {
                    let script = |extra: &str| {
                        let builder = format!("{body}().{chain}{extra}");
                        if place {
                            format!("place({builder},line(330.0,330.0));")
                        } else {
                            format!("create({builder},1);")
                        }
                    };
                    let omitted = run_script_with_seed(&script(""), 21).0;
                    let explicit = run_script_with_seed(&script(".tie(1)"), 21).0;
                    assert_eq!(format!("{omitted:?}"), format!("{explicit:?}"));
                    for value in [0., 0.5, 1.] {
                        for builder in [
                            format!("{body}().tie({value}).{chain}"),
                            format!("{body}().{chain}.tie({value})"),
                        ] {
                            let src = if place {
                                format!("place({builder},line(330.0,330.0));")
                            } else {
                                format!("create({builder},1);")
                            };
                            assert_eq!(
                                first_spawn_spec_for_script(&src).control.phonation.tie,
                                value
                            );
                        }
                    }
                }
            }
        }
    }
    for chain in [
        "sustain().metric().cycles(1).rest()",
        "rest().cycles(1).metric().sustain()",
    ] {
        let src = format!("create(sine().{chain}.pitch_apply_mode(\"gate_snap\"),1);");
        assert_eq!(first_spawn_spec_for_script(&src).control.phonation.tie, 1.);
    }
    for chain in [
        "sustain().entrained().pitch_apply_mode(\"gate_snap\")",
        "sustain().flow().pitch_apply_mode(\"rearticulate\")",
        "sustain().metric().anchor().pitch_apply_mode(\"gate_snap\")",
        "sustain().steady().pitch_apply_mode(\"gate_snap\")",
        "sustain().steady().pitch_apply_mode(\"rearticulate\")",
        "sustain().steady().pitch_apply_mode(\"glide\")",
        "sustain().metric().cycles(1)",
        "metric().while_alive()",
        "entrainment(0.5).sustain()",
        "sustain().entrainment(0.5).cycles(1).rest().pitch_apply_mode(\"gate_snap\")",
    ] {
        let script = |extra: &str| format!("create(sine().{chain}{extra},1);");
        let omitted = run_script_with_seed(&script(""), 21).0;
        let explicit = run_script_with_seed(&script(".tie(0)"), 21).0;
        assert_eq!(format!("{omitted:?}"), format!("{explicit:?}"));
        assert_eq!(
            first_spawn_spec_for_script(&script(""))
                .control
                .phonation
                .tie,
            0.
        );
    }
}

#[test]
fn synchronized_tie_default_sound_and_live_modes_use_existing_policy() {
    use crate::core::log2space::Log2Space;
    use crate::core::modulation::NeuralRhythms;
    use crate::life::voice::PhonationBatch;
    let tb = Timebase {
        fs: 48000.,
        hop: 512,
    };
    let spawn = |mode: &str, tie: &str| {
        let (scene, _) = run_script_with_seed(
            &format!(
                "create(sine().brain(\"drone\").sustain().metric().pitch_apply_mode(\"{mode}\"){tie},1);"
            ),
            21,
        );
        let mut c = Community::new(tb);
        c.set_seed(21);
        for action in scene.events.iter().flat_map(|e| &e.actions) {
            if matches!(action, Action::Spawn { .. }) {
                c.apply_action(action.clone(), &Landscape::default());
            }
        }
        c.voices.remove(0)
    };
    let space = Log2Space::new(55., 8000., 96);
    let rhythms = NeuralRhythms::default();
    let mut default = spawn("gate_snap", "");
    let mut one = spawn("gate_snap", ".tie(1)");
    let mut zero = spawn("gate_snap", ".tie(0)");
    let mut glide = spawn("glide", "");
    let mut glide_zero = spawn("glide", ".tie(0)");
    let mut counts = [0; 5];
    for i in 0..200 {
        let mut sounded = Vec::new();
        for (index, v) in [
            &mut default,
            &mut one,
            &mut zero,
            &mut glide,
            &mut glide_zero,
        ]
        .into_iter()
        .enumerate()
        {
            let mut out = PhonationBatch::default();
            v.prepare_phonation_batch(&mut out);
            v.tick_phonation_into_with_arrival(
                &tb,
                i * 512,
                &rhythms,
                None,
                0.,
                1.,
                0.8,
                None,
                Some(&space),
                &mut out,
            );
            counts[index] += out.onsets.len();
            sounded.push(format!("{:?}", out.onsets));
        }
        assert_eq!(sounded[0], sounded[1]);
        assert_eq!(sounded[3], sounded[4]);
    }
    assert_eq!(counts[0], 1);
    assert!(counts[2] > 1);
    assert_eq!(counts[2], counts[3]);
    for (segment, mode) in [
        PitchApplyMode::GateSnap,
        PitchApplyMode::Glide,
        PitchApplyMode::Rearticulate,
    ]
    .into_iter()
    .enumerate()
    {
        for v in [&mut default, &mut one, &mut glide] {
            v.apply_patch(&ControlUpdate {
                pitch_apply_mode: Some(mode),
                ..Default::default()
            })
            .unwrap();
        }
        for i in (200 + segment as u64 * 200)..(400 + segment as u64 * 200) {
            let mut sounded = Vec::new();
            for v in [&mut default, &mut one, &mut glide] {
                let mut out = PhonationBatch::default();
                v.prepare_phonation_batch(&mut out);
                v.tick_phonation_into_with_arrival(
                    &tb,
                    i * 512,
                    &rhythms,
                    None,
                    0.,
                    1.,
                    0.8,
                    None,
                    Some(&space),
                    &mut out,
                );
                sounded.push(format!("{:?}", out.onsets));
            }
            assert_eq!(sounded[0], sounded[1]);
            assert_eq!(sounded[0], sounded[2]);
        }
    }
}

#[test]
fn synchronized_tie_default_respawn_and_omissions_do_not_warn() {
    for chain in [
        "sustain().metric().pitch_apply_mode(\"gate_snap\")",
        "sustain().metric().pitch_apply_mode(\"rearticulate\")",
        "sustain().metric().pitch_apply_mode(\"glide\")",
        "sustain().metric().anchor().pitch_apply_mode(\"gate_snap\")",
        "sustain().steady().pitch_apply_mode(\"gate_snap\")",
        "sustain().steady().pitch_apply_mode(\"rearticulate\")",
        "sustain().steady().pitch_apply_mode(\"glide\")",
    ] {
        let (_, warnings) = run_script(&format!("create(sine().{chain},1);"));
        assert!(warnings.lock().unwrap().is_empty());
    }
    let (scene, _) = run_script_with_seed(
        "create(sine().sustain().metric().pitch_apply_mode(\"gate_snap\").respawn_random().respawn_background_death_rate(10000.0),1);",
        21,
    );
    let mut c = Community::new(Timebase {
        fs: 48000.,
        hop: 257,
    });
    c.set_seed(21);
    let landscape = Landscape::default();
    for action in scene.events.iter().flat_map(|e| &e.actions) {
        if matches!(
            action,
            Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
        ) {
            c.apply_action(action.clone(), &landscape);
        }
    }
    let parent = c.voices[0].id();
    c.advance(257, 0, 257. / 48000., &landscape);
    c.cleanup_dead(0, 257. / 48000., false, &landscape);
    let mut children = 0;
    for child in c.voices.iter_mut().filter(|v| v.id() != parent) {
        children += 1;
        assert_eq!(child.effective_control.phonation.tie, 1.);
        let shape = child.phonation_engine.clock.shape_mut().unwrap();
        assert!(!shape.has_tone);
        assert!(!shape.tie.as_ref().unwrap().pending_pitch);
        assert_eq!(shape.tie.as_ref().unwrap().trials, 0);
    }
    assert!(children > 0);
}

#[test]
fn tie_respawn_inherits_parameter_and_resets_private_policy() {
    use crate::life::phonation_engine::PhonationClock;
    use crate::life::shape_policy::Shape;
    let (scenario, _) = run_script_with_seed(
        "create(sine().sustain().entrained().pitch_apply_mode(\"rearticulate\").rearticulate_above(50).tie(0.5).respawn_random().respawn_background_death_rate(10000.0),1);",
        21,
    );
    let mut community = Community::new(Timebase {
        fs: 48000.,
        hop: 257,
    });
    community.set_seed(scenario.seed);
    let landscape = Landscape::default();
    for action in scenario.events.iter().flat_map(|event| &event.actions) {
        if matches!(
            action,
            Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
        ) {
            community.apply_action(action.clone(), &landscape);
        }
    }
    let parent = community.voices[0].id();
    let PhonationClock::Participation(clock) = &mut community.voices[0].phonation_engine.clock
    else {
        panic!()
    };
    let shape = clock.shape.as_mut().unwrap();
    shape.state = Shape::Connect;
    shape.veto_plan = true;
    shape.has_tone = true;
    shape.tie.as_mut().unwrap().pending_pitch = true;
    shape.tie.as_mut().unwrap().trials = 100;
    community.advance(257, 0, 257. / 48000., &landscape);
    community.cleanup_dead(0, 257. / 48000., false, &landscape);
    let children: Vec<_> = community
        .voices
        .iter()
        .filter(|v| v.id() != parent)
        .collect();
    assert!(!children.is_empty());
    for child in children {
        assert_eq!(child.effective_control.phonation.tie, 0.5);
        let PhonationClock::Participation(clock) = &child.phonation_engine.clock else {
            panic!()
        };
        let shape = clock.shape.as_ref().unwrap();
        assert_eq!(shape.state, Shape::Base);
        assert!(!shape.veto_plan && !shape.has_tone);
        assert!(!shape.tie.as_ref().unwrap().pending_pitch);
        assert_eq!(shape.tie.as_ref().unwrap().trials, 0);
    }
}

#[test]
fn explicit_pitch_and_viability_choices_override_preset_defaults_in_either_order() {
    for (a, b) in [
        (
            "glide(0.12).pitch_core(\"peak_sampler\").seek_consonance()",
            "seek_consonance().pitch_core(\"peak_sampler\").glide(0.12)",
        ),
        (
            "viability_scope(\"total\").consonance_viability(0.2, 0.8)",
            "consonance_viability(0.2, 0.8).viability_scope(\"total\")",
        ),
    ] {
        let a = first_spawn_spec_for_script(&format!("place(harmonic().{a}, line(220, 330));"));
        let b = first_spawn_spec_for_script(&format!("place(harmonic().{b}, line(220, 330));"));
        assert_eq!(a.control.pitch.core_kind, b.control.pitch.core_kind);
        assert_eq!(
            a.control.pitch.pitch_glide_tau_sec,
            b.control.pitch.pitch_glide_tau_sec
        );
        for spec in [&a, &b] {
            if let ArticulationCoreConfig::Entrain {
                lifecycle:
                    LifecycleConfig::Sustain {
                        selection_environment,
                        ..
                    },
                ..
            } = &spec.articulation
            {
                assert!(!selection_environment);
            }
        }
    }
    let explicit = first_spawn_spec_for_script(
        "place(harmonic().glide(0.12).pitch_core(\"peak_sampler\").seek_consonance(), line(220,330));",
    );
    assert_eq!(explicit.control.pitch.core_kind, PitchCoreKind::PeakSampler);
    assert_eq!(explicit.control.pitch.pitch_glide_tau_sec, 0.12);
    let implicit =
        first_spawn_spec_for_script("place(harmonic().seek_consonance(), line(220,330));");
    assert_eq!(implicit.control.pitch.pitch_glide_tau_sec, 0.30);
}

#[test]
fn counts_accept_integers_only_and_removed_verbs_are_unregistered() {
    for method in ["unison", "top_k", "random_candidates", "respawn_capacity"] {
        run_script(&format!("place(harmonic().{method}(2), at(220));"));
        assert!(
            run_script_err(&format!("harmonic().{method}(2.0);"))
                .message
                .contains(method)
        );
    }
    for script in [
        "variant(harmonic());",
        "harmonic().pitch_smooth(0.12);",
        "harmonic().selection_approx_loo(true);",
        "harmonic().leave_self_out_harmonics(2);",
        "let p = place(harmonic(), at(220)); p.leave_self_out_harmonics(2);",
        "set_control_update_mode(\"snapshot_phased\");",
        "set_scaffold_off();",
        "harmonic().respawn_min_c_level(0.5);",
    ] {
        run_script_err(script);
    }
}

#[test]
fn rest_on_habituation_is_initial_only_validated_and_order_independent() {
    for timing in ["metric()", "entrained()", "steady()", "flow()"] {
        for chain in [
            format!("sustain().{timing}.rest_on_habituation(0.25)"),
            format!("rest_on_habituation(0.25).{timing}.sustain()"),
        ] {
            let spec = first_spawn_spec_for_script(&format!("create(sine().{chain},1);"));
            assert_eq!(spec.control.phonation.rest_on_habituation, 0.25);
        }
    }
    for chain in [
        "sustain().rest_on_habituation(0.25)",
        "metric().rest_on_habituation(0.25)",
        "sustain().metric().rest_on_habituation(-1)",
        "sustain().metric().rest_on_habituation(1e100)",
    ] {
        let (scenario, warnings) = run_script(&format!("create(sine().{chain},1);"));
        assert!(
            warnings
                .lock()
                .unwrap()
                .iter()
                .any(|w| w.message.contains("rest_on_habituation"))
        );
        for action in scenario.events.iter().flat_map(|e| &e.actions) {
            if let Action::Spawn { spec, .. } = action {
                assert_eq!(spec.control.phonation.rest_on_habituation, 0.0);
            }
        }
    }
    assert!(
        run_script_err("let p=create(sine().sustain().metric(),1); p.rest_on_habituation(0.25);")
            .to_string()
            .contains("rest_on_habituation")
    );
    for value in ["0", "1", "0.25", "4", "1e30"] {
        let spec = first_spawn_spec_for_script(&format!(
            "create(sine().sustain().metric().rest_on_habituation({value}),1);"
        ));
        assert_eq!(
            spec.control.phonation.rest_on_habituation,
            value.parse::<f32>().unwrap()
        );
    }
}

#[test]
fn rest_on_habituation_respawn_resets_private_state_and_retains_weight() {
    let (scenario, _) = run_script_with_seed(
        "create(sine().sustain().entrained().cycles(2).rest(0.5).rest_on_habituation(0.25).pitch_apply_mode(\"gate_snap\").respawn_random().respawn_background_death_rate(10000.0),1);",
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
            Action::Spawn { .. } | Action::SetRespawnPolicy { .. }
        ) {
            community.apply_action(action.clone(), &landscape);
        }
    }
    let gap = community.voices[0]
        .phonation_engine
        .span_gap
        .as_mut()
        .unwrap();
    gap.own = 0.8;
    gap.exit_integral = 4.0;
    gap.return_integral = 3.0;
    let parent = community.voices[0].id();
    community.voices[0]
        .phonation_engine
        .rest
        .as_mut()
        .unwrap()
        .phase = crate::life::phonation_engine::RestPhase::Gap {
        origin: 0,
        until: None,
    };
    community.advance(257, 0, 257. / 48000., &landscape);
    community.cleanup_dead(0, 257. / 48000., false, &landscape);
    let children: Vec<_> = community
        .voices
        .iter()
        .filter(|v| v.id() != parent)
        .collect();
    assert!(!children.is_empty());
    assert!(community.rest_enabled);
    for child in children {
        let gap = child.phonation_engine.span_gap.as_ref().unwrap();
        assert_eq!(gap.weight, 0.25);
        assert_eq!(gap.phase, crate::life::span_gap::Phase::Unstarted);
        assert_eq!(
            (gap.own, gap.exit_integral, gap.return_integral),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(child.effective_control.phonation.rest_periods, Some(0.5));
        assert_eq!(
            child.effective_control.phonation.finite_duration,
            Some(DurationSpec::Gates(2))
        );
        assert_eq!(
            child.phonation_engine.rest.unwrap().phase,
            crate::life::phonation_engine::RestPhase::Ready
        );
    }
}

#[test]
fn dwell_is_initial_only_clamped_and_requires_explicit_free_seek() {
    for chain in [
        "dwell(0.75).seek_consonance()",
        "seek_consonance().dwell(0.75)",
    ] {
        let spec = first_spawn_spec_for_script(&format!("place(sine().{chain},at(220));"));
        assert_eq!(spec.control.pitch.dwell, 0.75);
    }
    for (chain, expected, warns) in [
        ("seek_consonance()", 0.0, false),
        ("seek_consonance().dwell(0)", 0.0, false),
        ("seek_consonance().dwell(1)", 1.0, false),
        ("seek_consonance().dwell(2)", 2.0, false),
        ("seek_consonance().dwell(1e100)", 0.0, true),
        ("seek_consonance().dwell(4)", 4.0, false),
        ("seek_consonance().dwell(1e30)", 1e30, false),
        ("seek_consonance().dwell(-1)", 0.0, true),
        ("dwell(0.5)", 0.0, true),
        ("seek_consonance().freq(220).dwell(0.5)", 0.0, true),
    ] {
        let (scenario, warnings) = run_script(&format!("create(sine().{chain},1);"));
        assert_eq!(
            warnings
                .lock()
                .unwrap()
                .iter()
                .any(|w| w.message.contains("dwell")),
            warns,
            "{chain}"
        );
        for action in scenario.events.iter().flat_map(|e| &e.actions) {
            if let Action::Spawn { spec, .. } = action {
                assert_eq!(spec.control.pitch.dwell, expected, "{chain}");
            }
        }
    }
    for invalid in [FLOAT::NAN, FLOAT::INFINITY, FLOAT::NEG_INFINITY] {
        let mut spec = PopulationSpec::preset(BodyMethod::Sine);
        spec.set_dwell(invalid);
        assert_eq!(spec.control.pitch.dwell, 0.0);
        assert!(
            spec.diagnostics
                .lock()
                .unwrap()
                .iter()
                .any(|w| w.message.contains("dwell"))
        );
    }
    assert!(
        run_script_err("let p=create(sine().seek_consonance(),1); p.dwell(1);")
            .to_string()
            .contains("dwell")
    );
}

#[test]
fn dwell_respawn_inherits_weight_without_phase() {
    let (scenario, _) = run_script_with_seed(
        "create(sine().seek_consonance().dwell(0.75).respawn_random().respawn_background_death_rate(10000.0),1);",
        21,
    );
    let mut community = Community::new(Timebase {
        fs: 48000.0,
        hop: 257,
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
    let parent = community.voices[0].id();
    community.advance(257, 0, 257.0 / 48000.0, &landscape);
    community.cleanup_dead(0, 257.0 / 48000.0, false, &landscape);
    let children: Vec<_> = community
        .voices
        .iter()
        .filter(|v| v.id() != parent)
        .collect();
    assert!(!children.is_empty());
    for child in children {
        assert_eq!(child.effective_control.pitch.dwell, 0.75);
        assert_eq!(child.accumulated_time_for_test(), 0.0);
        let state = child.pitch_ctl.dwell_state.unwrap();
        assert_eq!(
            (state.phase, state.remainder, state.elapsed_sec),
            (0.0, 0.0, 0.0)
        );
        assert_eq!(state.weight, 0.75);
    }
}

#[test]
fn own_sequence_initial_final_shape_and_invalid_weights() {
    let valid = "sustain().metric().seek_consonance().pitch_apply_mode(\"rearticulate\").rearticulate_above(0)";
    for (suffix, expected, warns) in [
        ("", 0.0, false),
        (".follow_own_sequence(0)", 0.0, false),
        (".follow_own_sequence(0.05)", 0.05, false),
        (".follow_own_sequence(1e100)", 1e100, false),
        (".follow_own_sequence(-1)", 0.0, true),
        (".follow_own_sequence(1).rearticulate_above(50)", 0.0, true),
        (
            ".follow_own_sequence(1).pitch_apply_mode(\"glide\")",
            0.0,
            true,
        ),
        (".follow_own_sequence(1).freq(220)", 0.0, true),
    ] {
        let (scenario, warnings) = run_script(&format!("place(sine().{valid}{suffix},at(220));"));
        assert_eq!(
            warnings
                .lock()
                .unwrap()
                .iter()
                .any(|w| w.message.contains("follow_own_sequence")),
            warns
        );
        for action in scenario.events.iter().flat_map(|e| &e.actions) {
            if let Action::Spawn { spec, .. } = action {
                assert_eq!(spec.control.pitch.follow_own_sequence, expected);
            }
        }
    }
    for value in [FLOAT::NAN, FLOAT::INFINITY, FLOAT::NEG_INFINITY] {
        let mut spec = PopulationSpec::preset(BodyMethod::Sine);
        spec.set_follow_own_sequence(value);
        assert_eq!(spec.control.pitch.follow_own_sequence, 0.0);
    }
    assert!(
        run_script_err("let p=create(sine(),1); p.follow_own_sequence(1);")
            .to_string()
            .contains("follow_own_sequence")
    );
}

#[test]
fn own_sequence_respawn_inherits_weight_without_learning() {
    let (scenario, _) = run_script_with_seed(
        "place(sine().sustain().steady().seek_consonance().pitch_apply_mode(\"rearticulate\").follow_own_sequence(0.75).respawn_random().respawn_background_death_rate(10000),at(220));",
        21,
    );
    let mut community = Community::new(Timebase {
        fs: 48000.0,
        hop: 512,
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
    let parent = community.voices[0].id();
    community.voices[0]
        .pitch_ctl
        .own_sequence
        .as_mut()
        .unwrap()
        .model
        .learn(1, 0.0)
        .unwrap();
    community.advance(512, 0, 512.0 / 48000.0, &landscape);
    community.cleanup_dead(0, 512.0 / 48000.0, false, &landscape);
    let children: Vec<_> = community
        .voices
        .iter()
        .filter(|v| v.id() != parent)
        .collect();
    assert!(!children.is_empty());
    for child in children {
        assert_eq!(child.effective_control.pitch.follow_own_sequence, 0.75);
        let sequence = child.pitch_ctl.own_sequence.as_ref().unwrap();
        assert_eq!(sequence.epoch, 0);
        assert_eq!(sequence.version, 0);
        assert!(sequence.model.symbols.is_empty());
    }
}

#[test]
fn own_sequence_resident_limit_keeps_execution_records() {
    let (scenario, _) = run_script(
        "place(sine().sustain().steady().seek_consonance().pitch_apply_mode(\"rearticulate\").follow_own_sequence(1),at(220).count(17));",
    );
    let mut community = Community::new(Timebase {
        fs: 48000.0,
        hop: 512,
    });
    let field = Landscape::default();
    for action in scenario.events.iter().flat_map(|e| &e.actions) {
        if matches!(action, Action::Spawn { .. }) {
            community.apply_action(action.clone(), &field);
        }
    }
    assert_eq!(community.voices.len(), 17);
    assert_eq!(
        community
            .voices
            .iter()
            .filter(|v| v.pitch_ctl.own_sequence.as_ref().unwrap().resident)
            .count(),
        16
    );
    let rejected = community.voices.last_mut().unwrap();
    let s = rejected.pitch_ctl.own_sequence.as_mut().unwrap();
    assert!(s.resource_stopped);
    assert_eq!(s.reason, Some("resident_state_resource_limit"));
    s.trace_enabled = true;
    let mut batch = crate::life::voice::PhonationBatch::default();
    rejected.prepare_phonation_batch(&mut batch);
    rejected.tick_phonation_into(
        &Timebase {
            fs: 48000.0,
            hop: 512,
        },
        0,
        &crate::core::modulation::NeuralRhythms::default(),
        None,
        0.0,
        1.0,
        0.8,
        &mut batch,
    );
    rejected.finish_sequence_execution(&batch, 512);
    let s = rejected.pitch_ctl.own_sequence.as_ref().unwrap();
    assert!(!s.executions.is_empty());
    assert!(s.model.symbols.is_empty());
}

#[test]
fn phrase_boundary_is_initial_only_validated_on_the_final_form() {
    for value in [FLOAT::NAN, FLOAT::INFINITY, FLOAT::NEG_INFINITY, -1.] {
        let mut spec = PopulationSpec::preset(BodyMethod::Sine);
        spec.set_phrase_boundary(value);
        assert_eq!(spec.control.phonation.phrase_boundary, 0.);
    }
    let mut spec = PopulationSpec::preset(BodyMethod::Sine);
    spec.set_phrase_boundary(f64::MAX);
    assert_eq!(spec.control.phonation.phrase_boundary, f64::MAX);
    for chain in [
        "phrase_boundary(1).sustain().steady().pitch_apply_mode(\"rearticulate\")",
        "steady().sustain().pitch_apply_mode(\"rearticulate\").phrase_boundary(1.0)",
    ] {
        let spec = first_spawn_spec_for_script(&format!("create(sine().{chain},1);"));
        assert_eq!(spec.control.phonation.phrase_boundary, 1.);
    }
    for chain in [
        "sine().phrase_boundary(1)",
        "sine().steady().sustain().pitch_apply_mode(\"glide\").phrase_boundary(1)",
        "harmonic().steady().sustain().phrase_boundary(1)",
        "sine().steady().sustain().phrase_boundary(-1)",
    ] {
        let spec = first_spawn_spec_for_script(&format!("create({chain},1);"));
        assert_eq!(spec.control.phonation.phrase_boundary, 0.);
    }
    assert!(
        run_script_err("let p=create(sine().steady().sustain(),1); p.phrase_boundary(1);")
            .to_string()
            .contains("phrase_boundary")
    );
}
