use super::*;

fn options() -> WiringOptions {
    WiringOptions {
        offline_body_probe: None,
        raw_t2_delay: Duration::ZERO,
        ui_tx: None,
        listener_forced: false,
        wait_user_exit: false,
        start_playing: true,
        audio_prod: None,
        wav_tx: None,
        reporter: None,
        deterministic_analysis: true,
        deterministic_footprints: true,
        guard_meter: None,
        underrun_frames: None,
        reserve_runtime_ids_through: 0,
        profile: None,
        audio_counters: None,
    }
}

fn fixture(
    mut opts: WiringOptions,
    source: &str,
) -> (RuntimeWiring, Arc<AtomicBool>, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "conchordal-q1-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&dir).unwrap();
    let script = dir.join("input.rhai");
    std::fs::write(&script, source).unwrap();
    let scenario = compile_scenario_from_script(&script, None).unwrap();
    let mut config = AppConfig::default();
    config.analysis.nfft = 2048;
    config.render_prototype = true;
    config.temporal_onset_comparison = Some(Default::default());
    if opts.reporter.is_none() {
        opts.reporter =
            Some(JsonlReporter::create(dir.join("run.jsonl").to_str().unwrap()).unwrap());
    }
    let stop = Arc::new(AtomicBool::new(false));
    let wiring = wire_runtime(
        &config,
        48_000,
        "Q1 fixture".into(),
        scenario,
        Arc::clone(&stop),
        opts,
    )
    .unwrap();
    (wiring, stop, dir)
}

fn join(wiring: RuntimeWiring) -> Result<(), String> {
    join_runtime(
        Some(wiring.worker_handle),
        [
            ("analysis", Some(wiring.analysis_handle)),
            ("listener", wiring.listener_analysis_handle),
        ],
    )
}

#[test]
fn final_report_is_written_once_while_waiting_for_user_exit() {
    let (tx, rx) = bounded(UI_CHANNEL_CAPACITY);
    let mut opts = options();
    opts.wait_user_exit = true;
    opts.ui_tx = Some(tx);
    opts.listener_forced = true;
    let (wiring, stop, dir) = fixture(opts, "seed(21); wait(0.03);");
    loop {
        if rx
            .recv_timeout(Duration::from_secs(10))
            .unwrap()
            .meta
            .playback_state
            == PlaybackState::Finished
        {
            break;
        }
    }
    let path = dir.join("run.jsonl");
    let deadline = Instant::now() + Duration::from_secs(5);
    let before = loop {
        let bytes = std::fs::read(&path).unwrap();
        if String::from_utf8_lossy(&bytes).contains("listener_confidence_summary") {
            break bytes;
        }
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    };
    thread::sleep(Duration::from_millis(60));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(!wiring.worker_handle.is_finished());
    stop.store(true, Ordering::SeqCst);
    join(wiring).unwrap();
    assert_eq!(std::fs::read(&path).unwrap(), before);
    let rows: Vec<serde_json::Value> = String::from_utf8(before)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    for kind in [
        "rhythm_summary",
        "listener_confidence_summary",
        "body_footprint_worker",
    ] {
        assert_eq!(
            rows.iter().filter(|row| row["type"] == kind).count(),
            1,
            "{kind}"
        );
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn multiple_worker_failures_return_without_an_error_receiver() {
    let mut opts = options();
    opts.reporter = Some(JsonlReporter::create("/dev/full").unwrap());
    opts.profile =
        Some(RunProfile::create("/dev/full", 21, true, 0.0, false, 48_000, 512, None).unwrap());
    opts.offline_body_probe = Some(Box::new(|state, _, _, _| {
        if state.failures.is_empty() {
            state.failures.push("raw T2 producer disconnected".into());
        }
    }));
    let (wiring, _, dir) = fixture(opts, "seed(21); wait(0.03);");
    let result = join(wiring).unwrap_err();
    assert!(result.contains("report"), "{result}");
    assert!(result.contains("profile"), "{result}");
    assert!(result.contains("raw T2 producer disconnected"), "{result}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn no_ui_consumer_builds_no_frames() {
    let mut opts = options();
    opts.offline_body_probe = Some(Box::new(|_, _, _, _| {
        UI_FRAME_BUILDS.with(|count| assert_eq!(count.get(), 0));
    }));
    let (wiring, _, dir) = fixture(opts, "seed(21); wait(0.03);");
    join(wiring).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn render_compile_errors_are_returned_before_opening_output() {
    let result = run_render(
        "invalid.txt",
        "/dev/full".into(),
        AppConfig::default(),
        Arc::new(AtomicBool::new(false)),
        None,
        None,
        0,
    );
    assert!(result.unwrap_err().contains(".rhai"));
}

#[test]
fn renderer_tail_does_not_bypass_ui_rate_limit_before_finish() {
    let (tx, rx) = bounded(UI_CHANNEL_CAPACITY);
    let mut opts = options();
    opts.ui_tx = Some(tx);
    let saw_tail = Arc::new(AtomicBool::new(false));
    let tail = Arc::clone(&saw_tail);
    opts.offline_body_probe = Some(Box::new(move |state, _, _, _| {
        if state.scenario_end_tick.is_some() && !state.schedule_renderer.is_idle() {
            tail.store(true, Ordering::SeqCst);
            UI_FRAME_BUILDS.with(|count| assert_eq!(count.get(), 1));
        }
        state.last_ui_update = Instant::now();
    }));
    let (wiring, _, dir) = fixture(
        opts,
        "seed(21); let v = place(sine().brain(\"drone\").sustain().anchor().amp(0.03).adsr(0.001, 0.0, 1.0, 0.2), at(220.0)); wait(0.1);",
    );
    join(wiring).unwrap();
    assert!(saw_tail.load(Ordering::SeqCst));
    let frames: Vec<_> = rx.try_iter().collect();
    assert_eq!(frames.len(), 2);
    assert_eq!(
        frames.last().unwrap().meta.playback_state,
        PlaybackState::Finished
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn private_body_worker_panic_fails_the_run_after_final_report_flush() {
    let mut opts = options();
    opts.listener_forced = true;
    let mut injected = false;
    opts.offline_body_probe = Some(Box::new(move |state, _, _, _| {
        if injected {
            return;
        }
        injected = true;
        let mut config = AppConfig::default();
        config.analysis.nfft = 2048;
        let core = build_analysis_runtime_core(&config, 48_000);
        let mut capture = crate::temporal_cognition::body::Capture::spawn(
            core.nsgt,
            crate::config::TemporalBodyConfig {
                means: [0.; 6],
                deviations: [1.; 6],
                accent_means: [0.; 2],
                accent_deviations: [1.; 2],
            },
            false,
            None,
        );
        capture.inject_worker_panic();
        state.body_snapshot = Some(Box::new(capture.snapshot()));
        state.schedule_renderer.body_capture = Some(capture);
    }));
    let (wiring, _, dir) = fixture(opts, "seed(21); wait(0.03);");
    let result = join(wiring).unwrap_err();
    assert!(
        result.contains("private body analysis worker failed"),
        "{result}"
    );
    let report = std::fs::read_to_string(dir.join("run.jsonl")).unwrap();
    assert!(report.contains("listener_confidence_summary"));
    let rows: Vec<serde_json::Value> = report
        .lines()
        .map(|row| serde_json::from_str(row).unwrap())
        .collect();
    assert!(
        rows.iter()
            .any(|row| row["type"] == "body_observation" && row["state"] == "failed")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn audio_callback_error_during_playback_fails_the_worker_result() {
    let counters = Arc::new(crate::audio::output::AudioCallbackCounters::default());
    let counter = Arc::clone(&counters);
    let mut opts = options();
    opts.listener_forced = true;
    opts.audio_counters = Some(counters);
    opts.offline_body_probe = Some(Box::new(move |_, _, _, _| {
        counter.callback_errors_total.store(1, Ordering::Relaxed);
    }));
    let (wiring, _, dir) = fixture(opts, "seed(21); wait(0.2);");
    let result = join(wiring).unwrap_err();
    assert!(
        result.contains("audio callback failed: 1 stream errors"),
        "{result}"
    );
    assert!(
        std::fs::read_to_string(dir.join("run.jsonl"))
            .unwrap()
            .contains("rhythm_summary")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn audio_callback_error_after_finalization_fails_the_worker_result() {
    let counters = Arc::new(crate::audio::output::AudioCallbackCounters::default());
    let mut opts = options();
    opts.listener_forced = true;
    opts.audio_counters = Some(Arc::clone(&counters));
    opts.wait_user_exit = true;
    let (wiring, stop, dir) = fixture(opts, "seed(21); wait(0.03);");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !std::fs::read_to_string(dir.join("run.jsonl"))
        .unwrap()
        .contains("listener_confidence_summary")
    {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    assert!(!wiring.worker_handle.is_finished());
    counters.callback_errors_total.store(1, Ordering::Relaxed);
    while !wiring.worker_handle.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let finished_without_user_exit = wiring.worker_handle.is_finished();
    stop.store(true, Ordering::SeqCst);
    let result = join(wiring).unwrap_err();
    assert!(finished_without_user_exit);
    assert!(
        result.contains("audio callback failed: 1 stream errors"),
        "{result}"
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn final_ui_snapshot_is_retried_after_a_full_channel_becomes_available() {
    use crate::temporal_cognition::observation::ObservationState;
    let (tx, rx) = bounded(UI_CHANNEL_CAPACITY);
    for _ in 0..UI_CHANNEL_CAPACITY {
        tx.try_send(UiFrame::default()).unwrap();
    }
    let mut opts = options();
    opts.ui_tx = Some(tx);
    opts.wait_user_exit = true;
    opts.listener_forced = true;
    let (wiring, stop, dir) = fixture(opts, "temporal_mode(\"observe\"); seed(21); wait(0.03);");
    let deadline = Instant::now() + Duration::from_secs(10);
    while !std::fs::read_to_string(dir.join("run.jsonl"))
        .unwrap()
        .contains("listener_confidence_summary")
    {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(5));
    }
    assert!(!wiring.worker_handle.is_finished());
    for _ in 0..UI_CHANNEL_CAPACITY {
        assert_eq!(
            rx.try_recv().unwrap().meta.playback_state,
            PlaybackState::NotStarted
        );
    }
    let frame = rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert_eq!(frame.meta.playback_state, PlaybackState::Finished);
    for observer in frame.temporal.iter() {
        assert_eq!(observer.state, ObservationState::Finished);
        assert!(observer.received_frames > 0);
    }
    stop.store(true, Ordering::SeqCst);
    join(wiring).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}
