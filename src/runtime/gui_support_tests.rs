use super::*;
use std::sync::Mutex;

fn options(ui_tx: Option<Sender<UiFrame>>) -> WiringOptions {
    WiringOptions {
        score: None,
        end_request: Arc::new(AtomicBool::new(false)),
        duration_sec: None,
        offline_body_probe: None,
        raw_t2_delay: Duration::ZERO,
        ui_tx,
        listener_forced: true,
        wait_user_exit: false,
        start_playing: true,
        audio_prod: None,
        wav_tx: None,
        reporter: None,
        deterministic_analysis: true,
        deterministic_footprints: false,
        guard_meter: None,
        underrun_frames: None,
        reserve_runtime_ids_through: 0,
        profile: None,
        telemetry: false,
        audio_counters: None,
    }
}

#[test]
fn gui_support_preserves_zero_weight_pcm_events_rng_and_clocks() {
    let root = std::env::temp_dir().join(format!("conchordal-gui-support-{}", std::process::id()));
    let _cleanup = test_output_directory::TestOutputDirectory::new(root.clone());
    std::fs::create_dir_all(&root).unwrap();
    let script = root.join("scene.rhai");
    std::fs::write(&script, r#"
seed(735);
place(sine().amp(0.02).sustain().leave_self_out(true), consonance(220, 880).count(2));
place(harmonic().amp(0.02).flow().cycles(3), at(330));
place(flora().amp(0.02).fade(0).tone("sine"), consonance(220, 880));
wait(0.5);
set_support_focus_preference(0); set_support_conflict_preference(0); set_related_novelty_preference(0);
follow_harmonic_relations(0); follow_harmonic_continuation(0);
place(sine().amp(0.02), consonance(220, 880).count(2));
wait(0.5);
"#).unwrap();
    let mut baseline = None;
    for (case, gui, enabled) in [
        ("baseline", true, false),
        ("display", true, true),
        ("report-only", false, false),
    ] {
        let rows = Arc::new(Mutex::new(Vec::new()));
        let captured = rows.clone();
        let probe: OfflineBodyProbe = Box::new(move |state, start, _, pcm| {
            assert!(state.harmonic_relations.is_none());
            assert!(!state.current_landscape.support_weights.active());
            if start >= 48_000 / 2 {
                assert_eq!(state.current_landscape.support.is_some(), enabled);
                assert!(
                    state
                        .current_landscape
                        .loo
                        .iter()
                        .all(|s| s.support.is_none())
                );
            }
            let voices: Vec<_> = state
                .pop
                .voices
                .iter()
                .map(|voice| {
                    use crate::life::phonation_engine::PhonationClock;
                    let clock = match &voice.phonation_engine.clock {
                        PhonationClock::ThetaGate(c) => format!("{c:?}"),
                        PhonationClock::Coupling(c) => format!("{c:?}"),
                        PhonationClock::Participation(c) => format!(
                            "{:?}",
                            (
                                c.hop_frames,
                                c.policy.as_ref().map(|p| p.rng_clock_state_for_test())
                            )
                        ),
                        PhonationClock::Custom(_) => panic!("fixture cannot use a custom clock"),
                    };
                    serde_json::json!({
                        "id":voice.id(), "pitch_rng":voice.pitch_ctl.rng_state_for_test(),
                        "sway_rng":format!("{:?}", voice.phonation_engine.sway_rng),
                        "articulation":format!("{:?}", voice.articulation), "clock":clock,
                    })
                })
                .collect();
            captured.lock().unwrap().push((
                pcm.map(|bus| bus.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
                serde_json::json!({"tick":start, "frame":state.frame_idx,
                    "community_rng":state.pop.rng_state_for_test(), "voices":voices,
                    "flora_rng":state.pop.flora.iter().map(|f| f.rng_state_for_test()).collect::<Vec<_>>() }),
            ));
        });
        let (tx, rx) = bounded::<UiFrame>(UI_CHANNEL_CAPACITY);
        let receiver = thread::spawn(move || {
            let mut seen = false;
            for frame in rx {
                seen |= frame.landscape.support.is_some();
            }
            seen
        });
        let report_path = root.join(format!("{case}.jsonl"));
        let mut opts = options(gui.then_some(tx));
        opts.offline_body_probe = Some(probe);
        opts.reporter = Some(JsonlReporter::create(report_path.to_str().unwrap()).unwrap());
        wiring::DISABLE_GUI_SUPPORT.set(!enabled);
        let wiring = wire_runtime(
            &AppConfig::default(),
            48_000,
            "GUI support".into(),
            compile_scenario_from_script(&script, Some(735)).unwrap(),
            Arc::new(AtomicBool::new(false)),
            opts,
        );
        wiring::DISABLE_GUI_SUPPORT.set(false);
        let wiring = wiring.unwrap();
        join_runtime(
            Some(wiring.worker_handle),
            [
                ("analysis", Some(wiring.analysis_handle)),
                ("listener", wiring.listener_analysis_handle),
            ],
        )
        .unwrap();
        assert_eq!(receiver.join().unwrap(), enabled);
        let records: Vec<serde_json::Value> = std::fs::read_to_string(report_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|r: &serde_json::Value| r["type"] != "hop_timing")
            .collect();
        assert!(!records.is_empty());
        let rows = Arc::try_unwrap(rows).unwrap().into_inner().unwrap();
        assert!(
            rows.iter()
                .any(|(pcm, _)| pcm.iter().flatten().any(|&v| v != 0))
        );
        if let Some((previous_rows, previous_records)) = &baseline {
            assert!(*previous_rows == rows, "PCM/RNG/clock mismatch: {case}");
            assert!(*previous_records == records, "event mismatch: {case}");
        } else {
            baseline = Some((rows, records));
        }
    }
}

#[test]
#[ignore = "GUI-4 paced ABBA acquisition requires the timing lock and L1 environment"]
fn gui_support_timing_acquisition() {
    let path = std::env::var("L1_SAMPLE").unwrap();
    let disabled = std::env::var("GUI4_SUPPORT").unwrap() == "off";
    let (tx, rx) = bounded::<UiFrame>(UI_CHANNEL_CAPACITY);
    let receiver = thread::spawn(move || {
        let mut supported = 0;
        let mut current = None;
        for frame in rx {
            if frame.landscape.support.is_some() {
                supported += 1;
            }
            current = Some(frame);
        }
        std::hint::black_box(current);
        supported
    });
    let mut opts = options(Some(tx));
    opts.duration_sec = Some(12.0);
    opts.deterministic_analysis = false;
    wiring::DISABLE_GUI_SUPPORT.set(disabled);
    let wiring = wire_runtime(
        &AppConfig::default(),
        48_000,
        path.clone(),
        compile_scenario_from_script(Path::new(&path), Some(21)).unwrap(),
        Arc::new(AtomicBool::new(false)),
        opts,
    );
    wiring::DISABLE_GUI_SUPPORT.set(false);
    let wiring = wiring.unwrap();
    join_runtime(
        Some(wiring.worker_handle),
        [
            ("analysis", Some(wiring.analysis_handle)),
            ("listener", wiring.listener_analysis_handle),
        ],
    )
    .unwrap();
    assert_eq!(receiver.join().unwrap() > 0, !disabled);
}
