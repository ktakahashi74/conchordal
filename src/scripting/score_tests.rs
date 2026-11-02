use super::score::*;
use super::*;
use crate::core::timebase::Timebase;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

fn timebase() -> Timebase {
    Timebase {
        fs: 48000.0,
        hop: 512,
    }
}

fn wait_until(mut predicate: impl FnMut() -> bool) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while !predicate() {
        assert!(
            Instant::now() < deadline,
            "score thread did not reach test barrier"
        );
        std::thread::yield_now();
    }
}

fn evaluate(
    source: &str,
    bound: f64,
    capacity: usize,
) -> (Vec<ScoreItem>, ScriptStatus, Result<(), String>) {
    let session = ScriptHost::start_source(source.into(), Some(42), Some(bound), capacity);
    let mut items = Vec::new();
    let result = session.drain(timebase(), &AtomicBool::new(false), |item| items.push(item));
    (items, session.stream.status(), result)
}

#[test]
fn bound_unwinds_nested_scopes_at_bound_before_finishing() {
    let (items, status, result) = evaluate(
        r#"
        endless();
        section("outer", || {
            let p = place(sine(), at(220));
            play(|| { let f = place(flora(), random(110, 880)); wait(100); });
            throw "must not execute";
        });
        throw "must not resume";
    "#,
        0.025,
        32,
    );
    result.unwrap();
    assert!(status.done);
    assert_eq!(status.evaluated_through, 0.025);
    let releases: Vec<_> = items
        .iter()
        .filter_map(|item| {
            if let ScoreItem::Event(event) = item {
                event
                    .actions
                    .iter()
                    .any(|a| {
                        matches!(
                            a,
                            Action::ReleasePopulation { .. } | Action::ReleaseFlora { .. }
                        )
                    })
                    .then_some(event.time)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(releases, vec![0.025, 0.025]);
}

#[test]
fn finished_and_failed_scores_drain_committed_items() {
    for suffix in ["", "throw \"expected failure\";"] {
        let (items, status, result) = evaluate(
            &format!("endless(); wait(0.02); place(sine(), at(220)); {suffix}"),
            1.0,
            8,
        );
        assert_eq!(items.len(), 1);
        assert_eq!(result.is_err(), !suffix.is_empty());
        assert_eq!(status.done, suffix.is_empty());
    }
}

#[test]
fn preheader_errors_and_closure_unblock_load_and_producer() {
    let session = ScriptHost::start_source("endless(); throw \"bad\";".into(), Some(1), None, 4);
    wait_until(|| session.stream.state.lock().unwrap().terminal.is_some());
    let state = session.stream.state.lock().unwrap();
    assert!(state.header.is_none());
    assert!(matches!(state.terminal, Some(Terminal::Failed(_))));
    drop(state);
    let looping = ScriptHost::start_source("endless(); loop {}".into(), Some(1), None, 4);
    looping.stream.close("stopped");
    drop(looping);
}

#[test]
fn emission_and_preheader_capacity_errors_are_distinct() {
    let (_, _, result) = evaluate("endless(); place(sine(), at(220));", 1.0, 1);
    assert!(result.unwrap_err().contains("score item exceeds"));
    let (_, _, result) = evaluate(
        "endless(); for i in 0..3 { place(sine(), at(220)); }",
        1.0,
        4,
    );
    assert!(result.unwrap_err().contains("before time advances exceed"));
}

#[test]
fn action_and_payload_overflow_have_deterministic_hop_times() {
    for source in [
        "endless(); wait(0.02); loop { place(sine(), at(220)); wait(0.00001); }".to_string(),
        format!(
            "endless(); wait(0.02); loop {{ section(\"{}\", || {{}}); wait(0.00001); }}",
            "x".repeat(size_of::<Action>() * 3)
        ),
    ] {
        let (_, _, a) = evaluate(&source, 1.0, 4);
        let (_, _, b) = evaluate(&source, 1.0, 4);
        let a = a.unwrap_err();
        assert!(a.contains("within one hop"), "{a}");
        assert_eq!(a, b.unwrap_err());
        assert!(a.contains("t=0.021333333"), "{a}");
    }
}

#[test]
fn pop_retry_consume_and_return_readiness_retry_keep_credit() {
    let stream = ScoreStream::new(None, 1);
    stream.set_header(ScriptContext::with_seed(1).scenario);
    let event = || {
        ScoreItem::Event(TimedEvent {
            time: 0.0,
            order: 1,
            actions: vec![Action::Finish],
        })
    };
    stream.push(event()).unwrap();
    stream.advance(1.0, false);
    let (_, charge) = stream.pop_due(0, timebase()).unwrap();
    let producer = stream.clone();
    let thread = std::thread::spawn(move || {
        producer.register_producer();
        producer.push(event()).unwrap();
    });
    wait_until(|| stream.state.lock().unwrap().pending.is_some());
    {
        let mut state = stream.state.lock().unwrap();
        state.worker_waiting = true;
        // A popped item remains charged until application has finished.
        assert_eq!(state.credit.actions, 0);
        assert_eq!(state.unreturned.actions, 1);
        assert!(!state.overflow());
    }
    stream.return_credit(charge);
    thread.join().unwrap();
    assert_eq!(stream.state.lock().unwrap().ring.len(), 1);

    // Hold the producer at a deterministic barrier after credit return.
    let (_, charge) = stream.pop_due(0, timebase()).unwrap();
    {
        let mut state = stream.state.lock().unwrap();
        state.pending = Some(charge);
        state.script_parked_full = true;
        state.watermark_sec = 0.0;
        state.worker_waiting = false;
    }
    stream.return_credit(charge);
    let p = stream.clone();
    let worker = std::thread::spawn(move || {
        p.wait_ready(
            0,
            timebase(),
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            false,
        )
    });
    wait_until(|| stream.state.lock().unwrap().worker_waiting);
    assert!(stream.state.lock().unwrap().terminal.is_none());
    stream.push(event()).unwrap();
    stream.advance(1.0, false);
    worker.join().unwrap().unwrap();
    assert!(stream.state.lock().unwrap().terminal.is_none());
}

#[test]
fn worker_transfers_credit_without_visiting_actions() {
    for count in [1, ACTION_CAPACITY] {
        let stream = ScoreStream::new(None, ACTION_CAPACITY);
        stream.set_header(ScriptContext::with_seed(1).scenario);
        RETAINED_ACTION_VISITS.with(|visits| visits.set(0));
        stream
            .push(ScoreItem::Event(TimedEvent {
                time: 0.0,
                order: 0,
                actions: vec![Action::Finish; count],
            }))
            .unwrap();
        assert_eq!(RETAINED_ACTION_VISITS.with(|visits| visits.get()), count);
        stream.advance(1.0, false);
        RETAINED_ACTION_VISITS.with(|visits| visits.set(0));
        let (item, charge) = stream.pop_due(0, timebase()).unwrap();
        assert_eq!(charge.actions, count);
        assert_eq!(charge.bytes, count * size_of::<Action>());
        assert_eq!(stream.state.lock().unwrap().unreturned, charge);
        drop(item);
        stream.return_credit(charge);
        assert_eq!(stream.state.lock().unwrap().credit, stream.capacity);
        assert_eq!(
            RETAINED_ACTION_VISITS.with(|visits| visits.get()),
            0,
            "worker must not scan even a full-capacity item"
        );
    }
}

#[test]
fn capacity_charges_allocation_capacity_and_close_discards() {
    let mut name = String::with_capacity(64);
    name.push('x');
    let item = ScoreItem::Marker(SceneMarker {
        time: 0.0,
        order: 1,
        name,
    });
    assert_eq!(item.retained_bytes(), 64);
    let stream = ScoreStream::new(None, 4);
    stream.push(item).unwrap();
    stream.close("stopped");
    stream
        .push(ScoreItem::Event(TimedEvent {
            time: 0.0,
            order: 2,
            actions: vec![Action::Finish],
        }))
        .unwrap();
    assert!(stream.state.lock().unwrap().ring.is_empty());
    assert!(stream.script_stop.load(Ordering::Acquire));
}

#[test]
fn pacing_publishes_header_before_runtime_and_can_be_cancelled() {
    let session = ScriptHost::start_source(
        "endless(); wait(100); place(sine(), at(220));".into(),
        Some(1),
        None,
        8,
    );
    wait_until(|| session.stream.state.lock().unwrap().parked_cursor_sec == 100.0);
    let state = session.stream.state.lock().unwrap();
    assert!(state.header.is_some());
    assert!(state.ring.is_empty());
    drop(state);
    session.stream.close("stopped");
    drop(session);
}

#[test]
fn bound_cannot_be_caught_and_warnings_are_coalesced() {
    let (items, status, result) = evaluate(
        r#"
        endless(); let p = place(sine(), at(220)); release(p); wait(0.02);
        loop { release(p); try { wait(0.02); } catch(e) { throw "caught bound"; } }
    "#,
        0.11,
        16,
    );
    result.unwrap();
    assert_eq!(items.len(), 2);
    assert_eq!(status.warnings.len(), 1);
    assert_eq!(status.warnings[0].count, 5);
}

#[test]
fn bundled_finite_scores_equal_incremental_scores_v3() {
    let mut files = Vec::new();
    for dir in ["samples", "samples/research", "tests/scripts"] {
        files.extend(
            std::fs::read_dir(dir)
                .unwrap()
                .map(|e| e.unwrap().path())
                .filter(|p| p.extension().is_some_and(|e| e == "rhai")),
        );
    }
    files.sort();
    let mut checked = 0;
    for file in files {
        let source = std::fs::read_to_string(&file).unwrap();
        if source.contains("endless(") || source.contains("parallel(") {
            continue;
        }
        let ctx = Arc::new(Mutex::new(ScriptContext::with_seed(42)));
        let engine = ScriptHost::create_engine(ctx.clone());
        if engine.eval::<Dynamic>(&source).is_err() {
            continue;
        }
        let mut ctx = ctx.lock().unwrap();
        ctx.finish();
        let eager = ctx.scenario.clone();
        let duration = eager.end.duration_sec().unwrap() + 1.0;
        let (streamed, _, result) =
            evaluate(&format!("endless();\n{source}"), duration, ACTION_CAPACITY);
        if let Err(error) = &result
            && error.contains("must be declared before time advances")
        {
            continue;
        }
        result.unwrap_or_else(|error| panic!("{}: {error}", file.display()));
        let mut expected: Vec<_> = eager
            .events
            .into_iter()
            .filter(|e| !e.actions.iter().any(|a| matches!(a, Action::Finish)))
            .map(ScoreItem::Event)
            .chain(eager.scene_markers.into_iter().map(ScoreItem::Marker))
            .collect();
        fn key(item: &ScoreItem) -> (f64, u64) {
            match item {
                ScoreItem::Event(e) => (e.time, e.order),
                ScoreItem::Marker(m) => (m.time, m.order),
            }
        }
        expected.sort_by(|a, b| {
            let (a, ao) = key(a);
            let (b, bo) = key(b);
            a.total_cmp(&b).then(ao.cmp(&bo))
        });
        fn normalize(mut items: Vec<ScoreItem>) -> String {
            for item in &mut items {
                if let ScoreItem::Event(e) = item {
                    for action in &mut e.actions {
                        if let Action::Spawn { ids, .. } = action {
                            *ids = crate::scenario::SpawnIds::Allocate(ids.len());
                        }
                    }
                }
            }
            format!("{items:?}")
        }
        assert_eq!(
            normalize(expected),
            normalize(streamed),
            "{}",
            file.display()
        );
        checked += 1;
    }
    eprintln!("V3 compared {checked} bundled scenarios");
    assert!(
        checked > 20,
        "bundled coverage unexpectedly small: {checked}"
    );
}

#[test]
fn header_freezes_final_basin_and_preserves_predeclaration_markers() {
    let source = r#"
        section("preface", || {}); endless(); seed(73); output_gain(-2);
        let p = place(sine().metric(), at(220)); temporal_basin(2,4);
        wait(1); place(sine().metric(), at(330)); wait(1);
    "#;
    let session = ScriptHost::start_source(source.into(), None, Some(3.0), 16);
    wait_until(|| session.stream.state.lock().unwrap().header.is_some());
    {
        let state = session.stream.state.lock().unwrap();
        let header = state.header.as_ref().unwrap();
        assert_eq!(header.seed, 73);
        assert_eq!(header.output_gain_db, -2.0);
        assert!(header.events.is_empty() && header.scene_markers.is_empty());
        assert_eq!(header.meter_shaping.basin_hz, Some((2.0, 4.0)));
    }
    let mut births = 0;
    let mut markers = 0;
    session
        .drain(timebase(), &AtomicBool::new(false), |item| match item {
            ScoreItem::Event(event) => {
                for action in event.actions {
                    if let Action::Spawn { spec, .. } = action {
                        if let PhonationTiming::Coupled(timing) = spec.control.phonation.spec.timing
                        {
                            assert_eq!(timing.base_rate_hz, 3.0);
                            births += 1;
                        } else {
                            panic!("expected metric founder");
                        }
                    }
                }
            }
            ScoreItem::Marker(marker) => {
                assert_eq!(marker.name, "preface");
                markers += 1;
            }
        })
        .unwrap();
    assert_eq!((births, markers), (2, 1));
}

#[test]
fn realtime_watchdog_and_human_end_close_waits() {
    let stream = ScoreStream::new(None, 4);
    let error = stream
        .wait_ready(
            0,
            Timebase {
                fs: 48000.0,
                hop: 1,
            },
            &AtomicBool::new(false),
            &AtomicBool::new(false),
            true,
        )
        .unwrap_err();
    assert!(error.contains("script fell behind"));
    assert!(stream.script_stop.load(Ordering::Acquire));
    let stream = ScoreStream::new(None, 4);
    stream
        .wait_ready(
            0,
            timebase(),
            &AtomicBool::new(false),
            &AtomicBool::new(true),
            false,
        )
        .unwrap();
    assert!(matches!(
        stream.state.lock().unwrap().terminal,
        Some(Terminal::Closed(_))
    ));
}

#[test]
fn oversized_scope_cleanup_fails_without_publishing_a_partial_release() {
    let (items, _, result) = evaluate(
        r#"
        endless(); section("scope", || {
            for i in 0..5 { place(sine(), at(220)); wait(0.1); }
        });
    "#,
        1.0,
        4,
    );
    assert!(
        result
            .unwrap_err()
            .contains("score item exceeds the ring capacity")
    );
    assert_eq!(
        items
            .iter()
            .filter(|item| matches!(item, ScoreItem::Event(_)))
            .count(),
        5
    );
    assert!(
        !items
            .iter()
            .any(|item| matches!(item, ScoreItem::Event(event)
        if event.actions.iter().any(|action| matches!(action, Action::ReleasePopulation { .. }))))
    );
}

#[test]
fn tonal_context_reservation_controls_endless_admission() {
    let voice = "sine().sustain().steady().seek_consonance().pitch_apply_mode(\"rearticulate\").follow_tonal_context(0.5)";
    for source in [
        format!("endless(); place({voice}, at(220)); wait(1);"),
        format!("endless(); wait(0.01); place({voice}, at(220));"),
        "endless(); wait(0.01); observe_tonal_context();".into(),
    ] {
        let session = ScriptHost::start_source(source, Some(21), Some(0.1), 32);
        let error = session
            .drain(timebase(), &AtomicBool::new(false), |_| {})
            .unwrap_err();
        assert!(error.contains("observe_tonal_context"), "{error}");
        let state = session.stream.state.lock().unwrap();
        let Some(Terminal::Failed(error)) = &state.terminal else {
            panic!("missing script error")
        };
        assert!(error.position.is_some_and(|p| p.line().is_some()));
    }
    let source = format!(
        "endless(); observe_tonal_context(); wait(0.01); place({voice}, at(220)); wait(0.02);"
    );
    let session = ScriptHost::start_source(source, Some(21), Some(0.1), 32);
    session
        .drain(timebase(), &AtomicBool::new(false), |_| {})
        .unwrap();
    assert!(
        session
            .stream
            .state
            .lock()
            .unwrap()
            .header
            .as_ref()
            .unwrap()
            .observe_tonal_context
    );
}

#[test]
fn finite_tonal_reservation_warns_without_reserving_unused_tables() {
    let session = ScriptHost::start_source(
        "observe_tonal_context(); place(sine(),at(220)); wait(0.01);".into(),
        Some(21),
        None,
        32,
    );
    session
        .drain(timebase(), &AtomicBool::new(false), |_| {})
        .unwrap();
    let state = session.stream.state.lock().unwrap();
    assert!(!state.header.as_ref().unwrap().observe_tonal_context);
    assert!(
        state
            .warnings
            .values()
            .any(|w| w.message.contains("ignored in a finite scenario")
                && w.position.line().is_some())
    );
}

#[test]
fn rain_scope_churn_releases_and_charges_mode_storage() {
    let (items, _, result) = evaluate(
        "endless(); loop { section(\"rain\", || {place(rain().fade(0.01), random(300,6000)); wait(0.01);}); }",
        0.03,
        32,
    );
    result.unwrap();
    let mut births = 0;
    let mut releases = 0;
    for item in items {
        if let ScoreItem::Event(event) = item {
            for action in event.actions {
                match &action {
                    Action::PlaceRain { .. } => {
                        births += 1;
                        assert!(action.heap_bytes() > 0);
                    }
                    Action::ReleaseRain { .. } => releases += 1,
                    _ => {}
                }
            }
        }
    }
    assert!(births > 1);
    assert_eq!(births, releases);
}

#[test]
fn endless_harmonic_patches_preserve_birth_membership_and_unrelated_controls() {
    let (items, _, result) = evaluate(
        r#"
        endless();
        let p=place(sine().sustain().seek_consonance().pitch_apply_mode("rearticulate").rearticulate_above(0), at(220));
        p.freq(330); p.amp(0.2); flush(); harmonic_rhythm(1); wait(0.01);
        p.pitch_apply_mode("glide"); p.rearticulate_above(20); p.amp(0.3); wait(0.01);
    "#,
        0.03,
        32,
    );
    result.unwrap();
    let mut amplitudes = Vec::new();
    for item in items {
        if let ScoreItem::Event(event) = item {
            for action in event.actions {
                if let Action::UpdatePopulation { patch, .. } = action {
                    assert!(patch.freq.is_none());
                    assert!(patch.pitch_apply_mode.is_none());
                    assert!(patch.rearticulate_above_cents.is_none());
                    amplitudes.extend(patch.amp);
                }
            }
        }
    }
    assert_eq!(amplitudes, [0.2, 0.3]);
}

#[test]
fn harmonic_reservation_controls_incremental_admission() {
    for control in ["follow_harmonic_relations", "follow_harmonic_continuation"] {
        for weight in ["1", "-0.5"] {
            let source = format!("endless(); wait(0.01); {control}({weight});");
            let session = ScriptHost::start_source(source, Some(21), Some(0.1), 32);
            let error = session
                .drain(timebase(), &AtomicBool::new(false), |_| {})
                .unwrap_err();
            assert!(error.contains("observe_harmonic_relations"), "{error}");
            let state = session.stream.state.lock().unwrap();
            let Some(Terminal::Failed(error)) = &state.terminal else {
                panic!("missing error")
            };
            assert!(error.position.is_some_and(|p| p.line().is_some()));
            drop(state);
            let source = format!(
                "endless(); observe_harmonic_relations(36); wait(0.01); {control}({weight}); wait(0.02);"
            );
            let session = ScriptHost::start_source(source, Some(21), Some(0.1), 32);
            session
                .drain(timebase(), &AtomicBool::new(false), |_| {})
                .unwrap();
            assert_eq!(
                session
                    .stream
                    .state
                    .lock()
                    .unwrap()
                    .header
                    .as_ref()
                    .unwrap()
                    .harmonic_relation_sources,
                Some(36)
            );
        }
    }
    for source in [
        "endless(); wait(0.01); observe_harmonic_relations(36);",
        "endless(); observe_harmonic_relations(0);",
        "endless(); observe_harmonic_relations(38401);",
    ] {
        assert!(
            evaluate(source, 0.1, 32)
                .2
                .unwrap_err()
                .contains("observe_harmonic_relations")
        );
    }
    let (_, _, result) = evaluate(
        "endless(); wait(0.01); follow_harmonic_relations(0); follow_harmonic_continuation(0.0); set_support_focus_preference(1);",
        0.1,
        32,
    );
    result.unwrap();
}

#[test]
fn finite_harmonic_reservation_keeps_event_prescan() {
    for declaration in ["", "observe_harmonic_relations(36);"] {
        let source = format!(
            "{declaration} wait(0.01); follow_harmonic_relations(1); follow_harmonic_continuation(-0.5); wait(0.02);"
        );
        let session = ScriptHost::start_source(source, Some(21), None, 32);
        session
            .drain(timebase(), &AtomicBool::new(false), |_| {})
            .unwrap();
        let state = session.stream.state.lock().unwrap();
        assert_eq!(
            state.header.as_ref().unwrap().harmonic_relation_sources,
            None
        );
        assert_eq!(
            state
                .warnings
                .values()
                .any(|w| w.kind == "finite_harmonic_observation" && w.position.line().is_some()),
            !declaration.is_empty()
        );
    }
}
