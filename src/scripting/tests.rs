use super::*;
use crate::core::landscape::LandscapeFrame;
use crate::core::timebase::Timebase;
use crate::life::population::Population;
use crate::life::voice::AnyArticulationCore;
use crate::life::voice::sound_body::SoundBody;
use crate::scenario::{DurationSpec, PhonationTiming, RhythmCouplingMode, RhythmRole};
use rand::SeedableRng;
use std::collections::HashMap;

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

const E4_STEP_RESPONSE_SCRIPT: &str = r#"
        let anchor = harmonic().anchor();
        let probe = harmonic().pitch_core("peak_sampler");

        section("E4 Step Response Test", || {
            let base = create(anchor, 1).freq(196.0);
            flush();

            let probes = create(probe, 12)
                .place(peaks(196.0).range(0.8, 2.5).spacing(0.9));
            flush();

            harmonic_tension(0.0);
            flush();
            harmonic_tension(0.5);
            flush();
            harmonic_tension(1.0);
            flush();

            release(probes);
            release(base);
            flush();
        });
    "#;

const E4_BETWEEN_RUNS_SCRIPT: &str = r#"
        let anchor = harmonic().anchor();
        let probe = harmonic().pitch_core("peak_sampler");

        section("E4 Between Runs Test", || {
            let weights = [0.0, 0.5, 1.0];
            for w in weights {
                harmonic_tension(w);
                flush();

                let base = create(anchor, 1).freq(196.0);
                let probes = create(probe, 8)
                    .place(peaks(196.0).range(0.8, 2.5).spacing(0.9));
                flush();

                release(probes);
                release(base);
                flush();
            }
        });
    "#;

#[test]
fn draft_group_without_commit_is_dropped() {
    let (scenario, warnings) = run_script(
        r#"
            create(sine(), 1);
        "#,
    );
    let has_spawn = scenario
        .events
        .iter()
        .any(|ev| ev.actions.iter().any(|a| matches!(a, Action::Spawn { .. })));
    assert!(!has_spawn, "draft should not spawn without wait/flush");
    assert_eq!(warnings.draft_dropped, 1);
}

#[test]
fn flush_spawns_without_advancing_time() {
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
fn scene_scope_releases_live_groups() {
    let (scenario, _warnings) = run_script(
        r#"
            section("alpha", || {
                let g = create(sine(), 1);
                flush();
                wait(0.5);
            });
        "#,
    );
    let mut release_time = None;
    for (time, action) in action_times(&scenario) {
        if matches!(action, Action::ReleaseGroup { .. }) {
            release_time = Some(time);
        }
    }
    assert_eq!(release_time, Some(0.5));
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
            if let Action::ReleaseGroup { fade_sec, .. } = action {
                release_tail = release_tail.max(event.time + fade_sec);
            }
        }
    }
    let expected = release_tail.max(1.0);
    assert!(matches!(finish_time, Some(t) if (t - expected).abs() <= 1e-6));
}

#[test]
fn scope_drop_warns_on_draft() {
    let (_scenario, warnings) = run_script(
        r#"
            section("alpha", || { create(sine(), 1); });
        "#,
    );
    assert_eq!(warnings.draft_dropped, 1);
}

#[test]
fn spawn_order_is_group_id_order() {
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
            if let Action::Spawn { group_id, ids, .. } = action {
                spawns.push((event.time, *group_id, ids.clone()));
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
fn place_material_builds_draft_participant_until_flush() {
    let (scenario, _warnings) = run_script(
        r#"
            let p = place(
                harmonic().amp(0.04).send(habitat_bus | presentation_bus),
                density(90.0, 900.0).count(3).spacing(0.8)
            );
            p.amp(0.02);
            flush();
        "#,
    );
    let mut spawn = None;
    for event in &scenario.events {
        for action in &event.actions {
            if let Action::Spawn {
                ids,
                spec,
                strategy,
                ..
            } = action
            {
                spawn = Some((
                    event.time,
                    ids.len(),
                    spec.control.clone(),
                    strategy.clone(),
                ));
            }
        }
    }
    let (time, count, control, strategy) = spawn.expect("spawn action");
    assert_eq!(time, 0.0);
    assert_eq!(count, 3);
    assert_eq!(control.body.amp, 0.02);
    assert!(control.body.routing.to_habitat);
    assert!(control.body.routing.to_presentation);
    assert!(matches!(
        strategy,
        Some(SpawnStrategy::ConsonanceDensity { .. })
    ));
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
    let control = scenario
        .events
        .iter()
        .flat_map(|event| &event.actions)
        .find_map(|action| match action {
            Action::Spawn { spec, .. } => Some(spec.control.clone()),
            _ => None,
        })
        .expect("spawn action");

    assert!(control.body.routing.to_habitat);
    assert!(!control.body.routing.to_presentation);
    assert!((control.pitch.freq - 330.0).abs() <= 1e-6);
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
fn place_then_freq_clears_strategy() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 4).place(peaks(220.0)).freq(330.0);
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
    assert!(strategy.is_none());
}

#[test]
fn freq_then_place_sets_strategy() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 4).freq(330.0).place(peaks(220.0));
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
    assert!(matches!(strategy, Some(SpawnStrategy::Consonance { .. })));
}

#[test]
fn place_then_anchor_locks_spawn_mode() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 4).place(peaks(220.0)).anchor();
            flush();
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
    assert!(matches!(strategy, Some(SpawnStrategy::Consonance { .. })));
    assert_eq!(mode, PitchMode::Lock);
}

#[test]
fn place_then_anchor_survives_spawn() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 2).place(peaks(220.0)).anchor();
            flush();
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

    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    pop.apply_action(spawn_action, &landscape, None);

    let voice = pop.voices.first().expect("spawned");
    assert_eq!(voice.base_control.pitch.mode, PitchMode::Lock);
    assert_eq!(voice.effective_control.pitch.mode, PitchMode::Lock);
}

#[test]
fn place_preserves_default_pitch_mode() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 4).place(peaks(220.0));
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
    assert_eq!(mode, PitchMode::Free);
}

#[test]
fn anchor_then_place_preserves_lock() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine(), 4).anchor().place(peaks(220.0));
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
fn species_anchor_sets_spawn_mode() {
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
fn consonance_movement_sets_free_hill_climb_glide_defaults() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().seek_consonance(), 1);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
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
fn species_pitch_core_sets_spawn_core_kind() {
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
fn species_landscape_weight_sets_spawn_control() {
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
fn species_landscape_weight_reaches_spawned_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().landscape_weight(0.3), 1);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.effective_control.pitch.landscape_weight - 0.3).abs() <= 1e-6);
}

#[test]
fn species_temperature_reaches_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().temperature(0.3), 1);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().temperature_for_test() - 0.3).abs() <= 1e-6);
}

#[test]
fn group_landscape_weight_emits_live_update() {
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
        if let Action::UpdateGroup { patch, .. } = action
            && patch.landscape_weight == Some(0.4)
        {
            saw_update = true;
            break;
        }
    }
    assert!(saw_update, "expected landscape_weight live update");
}

#[test]
fn group_landscape_weight_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.landscape_weight(0.6);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                pop.apply_action(action.clone(), &landscape, None);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.effective_control.pitch.landscape_weight - 0.6).abs() <= 1e-6);
}

#[test]
fn group_amp_live_update_preserves_member_pitch_centers() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 4).place(peaks(196.0).range(0.8, 2.5).spacing(0.9));
            flush();
            let g = g.amp(0.33);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    let mut before_update: Vec<(u64, f32)> = Vec::new();
    let mut after_update: Vec<(u64, f32)> = Vec::new();

    for event in &scenario.events {
        for action in &event.actions {
            pop.apply_action(action.clone(), &landscape, None);
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
            .any(|action| matches!(action, Action::UpdateGroup { .. }))
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
fn group_live_update_last_write_wins_within_flush() {
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
            Action::UpdateGroup { patch, .. } => Some(patch.clone()),
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
fn flush_emits_update_before_release_for_same_group() {
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
                .any(|action| matches!(action, Action::UpdateGroup { .. }))
                && event
                    .actions
                    .iter()
                    .any(|action| matches!(action, Action::ReleaseGroup { .. }))
        })
        .expect("event with both update and release");

    let update_idx = event
        .actions
        .iter()
        .position(|action| matches!(action, Action::UpdateGroup { .. }))
        .expect("update action");
    let release_idx = event
        .actions
        .iter()
        .position(|action| matches!(action, Action::ReleaseGroup { .. }))
        .expect("release action");
    assert!(
        update_idx < release_idx,
        "flush must emit update before release within same event"
    );
}

#[test]
fn group_temperature_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.temperature(0.4);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                pop.apply_action(action.clone(), &landscape, None);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().temperature_for_test() - 0.4).abs() <= 1e-6);
}

#[test]
fn species_crowding_sets_spawn_control() {
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
fn species_crowding_reaches_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().avoid_neighbors(1.2, 35.0), 1);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().crowding_strength_for_test() - 1.2).abs() <= 1e-6);
    assert!((voice.pitch_core_for_test().crowding_sigma_cents_for_test() - 35.0).abs() <= 1e-3);
}

#[test]
fn species_crowding_single_arg_uses_default_sigma() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().avoid_neighbors(0.8), 1);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().crowding_strength_for_test() - 0.8).abs() <= 1e-6);
    assert!((voice.pitch_core_for_test().crowding_sigma_cents_for_test() - 60.0).abs() <= 1e-3);
}

#[test]
fn species_crowding_mixed_numeric_overloads_work() {
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
fn group_crowding_live_update_reaches_individual_core() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.avoid_neighbors(0.8, 25.0);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                pop.apply_action(action.clone(), &landscape, None);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().crowding_strength_for_test() - 0.8).abs() <= 1e-6);
    assert!((voice.pitch_core_for_test().crowding_sigma_cents_for_test() - 25.0).abs() <= 1e-3);
}

#[test]
fn group_crowding_mixed_numeric_overloads_work() {
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
        if let Action::UpdateGroup { patch, .. } = action
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
fn group_crowding_target_emits_actions_for_draft_and_live_updates() {
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
        if let Action::SetGroupCrowdingTarget {
            same_group_visible,
            other_group_visible,
            ..
        } = action
        {
            if *same_group_visible && !*other_group_visible {
                saw_spawn_target = true;
            }
            if *same_group_visible && *other_group_visible {
                saw_live_target = true;
            }
        }
    }
    assert!(
        saw_spawn_target,
        "expected draft crowding_target to be emitted"
    );
    assert!(
        saw_live_target,
        "expected live crowding_target update to be emitted"
    );
}

#[test]
fn species_leave_self_out_and_temperature_reach_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().leave_self_out(true).temperature(0.12), 1);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!(voice.pitch_core_for_test().leave_self_out_for_test());
    assert!((voice.pitch_core_for_test().temperature_for_test() - 0.12).abs() <= 1e-6);
}

#[test]
fn group_leave_self_out_and_temperature_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.leave_self_out(true).temperature(0.2);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                pop.apply_action(action.clone(), &landscape, None);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!(voice.pitch_core_for_test().leave_self_out_for_test());
    assert!((voice.pitch_core_for_test().temperature_for_test() - 0.2).abs() <= 1e-6);
}

#[test]
fn species_move_cost_reaches_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(sine().move_cost(0.9), 1);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().move_cost_coeff_for_test() - 0.9).abs() <= 1e-6);
}

#[test]
fn group_move_cost_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g.move_cost(0.8);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                pop.apply_action(action.clone(), &landscape, None);
            }
            _ => {}
        }
    }
    let voice = pop.voices.first().expect("spawned");
    assert!((voice.pitch_core_for_test().move_cost_coeff_for_test() - 0.8).abs() <= 1e-6);
}

#[test]
fn group_hill_climb_knobs_and_exact_loo_live_update_reach_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g
                .neighbor_step_cents(25)
                .tessitura_gravity(0.12)
                .move_cost_exp(2)
                .leave_self_out(true)
                .leave_self_out_mode("exact");
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                pop.apply_action(action.clone(), &landscape, None);
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
    assert_eq!(
        core.leave_self_out_mode_for_test(),
        LeaveSelfOutMode::ExactScan
    );
    assert_eq!(
        voice.effective_control.pitch.leave_self_out_mode,
        LeaveSelfOutMode::ExactScan
    );
}

#[test]
fn species_peak_sampler_knobs_reach_spawned_core() {
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
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
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
fn species_advanced_pitch_knobs_reach_spawned_core() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                sine()                    .proposal_interval(0.3)
                    .global_peaks(12, 40.0)
                    .ratio_candidates(5)
                    .move_cost_time_scale("proposal_interval")
                    .leave_self_out_harmonics(4)
                    .pitch_apply_mode("glide")
                    .glide(0.08),
                1
            );
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        if let Action::Spawn { .. } = action {
            pop.apply_action(action.clone(), &landscape, None);
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
    assert_eq!(core.leave_self_out_harmonics_for_test(), 4);
    assert_eq!(
        voice.effective_control.pitch.pitch_apply_mode,
        PitchApplyMode::Glide
    );
    assert!((voice.effective_control.pitch.pitch_glide_tau_sec - 0.08).abs() <= 1e-6);
}

#[test]
fn group_advanced_pitch_knobs_live_update_reaches_individual() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1);
            flush();
            let g = g
                .proposal_interval(0.25)
                .global_peaks(10, 30)
                .ratio_candidates(4)
                .move_cost_time_scale("proposal_interval")
                .leave_self_out_harmonics(3)
                .pitch_apply_mode("glide")
                .glide(0.05);
            flush();
        "#,
    );
    let mut pop = Population::new(Timebase {
        fs: 48_000.0,
        hop: 64,
    });
    let landscape = LandscapeFrame::default();
    for action in scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
    {
        match action {
            Action::Spawn { .. } | Action::UpdateGroup { .. } => {
                pop.apply_action(action.clone(), &landscape, None);
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
    assert_eq!(core.leave_self_out_harmonics_for_test(), 3);
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
            set_pitch_objective("negative_consonance");
            wait(0.1);
            set_pitch_objective("consonance");
        "#,
    );
    let modes: Vec<PitchObjectiveMode> = scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
        .filter_map(|action| match action {
            Action::SetHarmonicityParams { update } => update.pitch_objective_mode,
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
fn reject_targets_wraps_spawn_strategy() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1)
                .place(reject_targets(random(200.0, 400.0), 220, [0, 7, 12], 0.35, 16));
            flush();
        "#,
    );
    let strategy = scenario
        .events
        .iter()
        .flat_map(|event| event.actions.iter())
        .find_map(|action| match action {
            Action::Spawn {
                strategy: Some(strategy),
                ..
            } => Some(strategy.clone()),
            _ => None,
        })
        .expect("spawn strategy");
    match strategy {
        SpawnStrategy::RejectTargets {
            base,
            anchor_hz,
            targets_st,
            exclusion_st,
            max_tries,
        } => {
            assert!(matches!(*base, SpawnStrategy::RandomLog { .. }));
            assert!((anchor_hz - 220.0).abs() <= 1e-6);
            assert_eq!(targets_st, vec![0.0, 7.0, 12.0]);
            assert!((exclusion_st - 0.35).abs() <= 1e-6);
            assert_eq!(max_tries, 16);
        }
        other => panic!("expected RejectTargets strategy, got {other:?}"),
    }
}

#[test]
fn group_draft_landscape_weight_sets_spawn_control() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1).landscape_weight(0.4);
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
    assert!((weight - 0.4).abs() <= 1e-6);
}

#[test]
fn species_respawn_policy_emits_runtime_action() {
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
            group_id, policy, ..
        } = action
        {
            assert_eq!(*group_id, 1);
            assert_eq!(*policy, RespawnPolicy::Hereditary { sigma_oct: 0.03 });
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy action");
}

#[test]
fn group_draft_respawn_random_emits_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1).respawn_random();
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
            group_id, policy, ..
        } = action
        {
            assert_eq!(*group_id, 1);
            assert_eq!(*policy, RespawnPolicy::Random);
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy(Random)");
}

#[test]
fn group_draft_respawn_hereditary_emits_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1).respawn_hereditary(0.03);
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
            group_id, policy, ..
        } = action
        {
            assert_eq!(*group_id, 1);
            assert_eq!(*policy, RespawnPolicy::Hereditary { sigma_oct: 0.03 });
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy(Hereditary)");
}

#[test]
fn group_draft_respawn_consonance_emits_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1).respawn_consonance();
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
            group_id,
            policy: RespawnPolicy::PeakBiased { config },
            ..
        } = action
        {
            assert_eq!(*group_id, 1);
            assert_eq!(*config, RespawnPeakBiasConfig::default());
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy(PeakBiased)");
}

#[test]
fn group_respawn_tier2_settings_reach_runtime_action() {
    let (scenario, _warnings) = run_script(
        r#"
            let g = create(sine(), 1)
                .respawn_hereditary(0.03)
                .respawn_settle(peaks(220.0).range(0.75, 1.5).spacing(0.5))
                .respawn_capacity(3)
                .respawn_min_c_level(0.4)
                .respawn_background_death_rate(0.03);
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
            group_id,
            policy,
            settle_strategy,
            capacity,
            min_c_level,
            background_death_rate_per_sec,
        } = action
        {
            assert_eq!(*group_id, 1);
            assert_eq!(*policy, RespawnPolicy::Hereditary { sigma_oct: 0.03 });
            assert_eq!(*capacity, 3);
            assert_eq!(*min_c_level, Some(0.4));
            assert!((*background_death_rate_per_sec - 0.03).abs() <= 1e-6);
            assert!(matches!(
                settle_strategy,
                Some(SpawnStrategy::Consonance {
                    root_freq,
                    min_mul,
                    max_mul,
                    min_dist_erb,
                }) if (*root_freq - 220.0).abs() <= 1e-6
                    && (*min_mul - 0.75).abs() <= 1e-6
                    && (*max_mul - 1.5).abs() <= 1e-6
                    && (*min_dist_erb - 0.5).abs() <= 1e-6
            ));
            saw_policy = true;
        }
    }
    assert!(saw_policy, "expected SetRespawnPolicy with tier-2 settings");
}

#[test]
fn set_control_update_mode_updates_scenario() {
    let (scenario, _warnings) = run_script(
        r#"
            set_control_update_mode("sequential_rotating");
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
            set_scaffold_shared(2.5);
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
            set_scaffold_scrambled(3.0, 17);
        "#,
    );
    assert!(matches!(
        scenario.scaffold,
        ScaffoldConfig::Scrambled { freq_hz, seed }
            if (freq_hz - 3.0).abs() <= 1e-6 && seed == 17
    ));
}

#[test]
fn spawn_payload_preserves_species_control_fields() {
    let (scenario, _warnings) = run_script(
        r#"
            create(harmonic(), 1)
                .amp(0.33)
                .freq(330.0)
                .brightness(0.7);
            flush();
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
                harmonic()                    .metabolism(0.1)
                    .viability_rate(0.3)
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
        selection_approx_loo,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert_eq!(continuous_recharge_score_low, Some(0.3));
    assert_eq!(continuous_recharge_score_high, Some(0.8));
    assert!(selection_approx_loo);
}

#[test]
fn spawn_payload_preserves_selection_approx_loo() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                harmonic()                    .metabolism(0.1)
                    .selection_approx_loo(true),
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
        selection_approx_loo,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert!(selection_approx_loo);
}

#[test]
fn selection_approx_loo_can_override_consonance_viability_for_reference_assays() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                harmonic()                    .metabolism(0.1)
                    .consonance_viability(0.3, 0.8)
                    .selection_approx_loo(false),
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
        selection_approx_loo,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert!(!selection_approx_loo);
}

#[test]
fn viability_scope_total_overrides_environment_default() {
    let (scenario, _warnings) = run_script(
        r#"
            create(
                harmonic()                    .metabolism(0.1)
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
        selection_approx_loo,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert!(!selection_approx_loo);
}

#[test]
fn draft_group_viability_scope_total_overrides_environment_default() {
    let (scenario, _warnings) = run_script(
        r#"
            create(harmonic(), 1)
                .metabolism(0.1)
                .consonance_viability(0.3, 0.8)
                .viability_scope("total");
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
        selection_approx_loo,
        ..
    } = lifecycle
    else {
        panic!("expected sustain lifecycle");
    };
    assert!(!selection_approx_loo);
}

#[test]
fn live_group_brightness_emits_timbre_patch() {
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
            Action::UpdateGroup { patch, .. } => Some(patch.clone()),
            _ => None,
        })
        .expect("update action");
    assert_eq!(patch.timbre_brightness, Some(0.25));
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
fn live_group_spread_and_unison_emit_timbre_patch() {
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
            Action::UpdateGroup { patch, .. } => Some(patch.clone()),
            _ => None,
        })
        .expect("update action");
    assert_eq!(patch.timbre_spread, Some(0.3));
    assert_eq!(patch.timbre_unison, Some(4));
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
fn e4_step_response_script_has_fixed_population_spawns() {
    let script = E4_STEP_RESPONSE_SCRIPT;
    assert!(
        script.contains("let anchor = harmonic()") && script.contains("let probe = harmonic()"),
        "E4 step response must use harmonic bodies"
    );
    assert!(
        script.contains(".anchor()"),
        "E4 step response anchor must lock pitch"
    );
    let (scenario, _warnings) = run_script(script);

    let mut spawn_actions = 0usize;
    let mut spawned_agents = 0usize;
    let mut mirror_updates = 0usize;
    for action in scenario.events.iter().flat_map(|ev| ev.actions.iter()) {
        match action {
            Action::Spawn { ids, .. } => {
                spawn_actions += 1;
                spawned_agents += ids.len();
            }
            Action::SetHarmonicityParams { update } if update.mirror.is_some() => {
                mirror_updates += 1;
            }
            _ => {}
        }
    }

    // One anchor group + one probe group should spawn once each.
    assert_eq!(spawn_actions, 2);
    assert_eq!(spawned_agents, 13);
    assert!(spawn_actions < mirror_updates);
}

#[test]
fn e4_between_runs_script_pairs_spawns_and_releases_per_weight() {
    let script = E4_BETWEEN_RUNS_SCRIPT;
    assert!(
        script.contains("let anchor = harmonic()") && script.contains("let probe = harmonic()"),
        "E4 between-runs must use harmonic bodies"
    );
    assert!(
        script.contains(".anchor()"),
        "E4 between-runs anchor must lock pitch"
    );
    let (scenario, _warnings) = run_script(script);

    let mut mirror_updates = 0usize;
    let mut spawn_by_group: HashMap<u64, usize> = HashMap::new();
    let mut release_by_group: HashMap<u64, usize> = HashMap::new();

    for action in scenario.events.iter().flat_map(|ev| ev.actions.iter()) {
        match action {
            Action::SetHarmonicityParams { update } if update.mirror.is_some() => {
                mirror_updates += 1;
            }
            Action::Spawn { group_id, .. } => {
                *spawn_by_group.entry(*group_id).or_insert(0) += 1;
            }
            Action::ReleaseGroup { group_id, .. } => {
                *release_by_group.entry(*group_id).or_insert(0) += 1;
            }
            _ => {}
        }
    }

    assert_eq!(spawn_by_group.len(), mirror_updates * 2);
    assert_eq!(release_by_group.len(), mirror_updates * 2);
    for (group_id, spawn_count) in &spawn_by_group {
        assert_eq!(*spawn_count, 1, "group {group_id} spawned more than once");
        assert_eq!(
            release_by_group.get(group_id).copied().unwrap_or(0),
            1,
            "group {group_id} does not have exactly one matching release"
        );
    }
}
