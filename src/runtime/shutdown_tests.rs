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

#[test]
#[ignore = "L-1 acquisition requires the machine timing lock"]
fn l1_sample_acquisition() {
    let path = std::env::var("L1_SAMPLE").expect("sample path");
    let scenario = compile_scenario_from_script(std::path::Path::new(&path), Some(21)).unwrap();
    let events: Vec<_> = scenario
        .events
        .iter()
        .map(|event| {
            serde_json::json!({
                "time_sec": event.time,
                "actions": event.actions.iter().map(ToString::to_string).collect::<Vec<_>>()
            })
        })
        .collect();
    serde_json::to_writer(
        std::fs::File::create(
            std::path::Path::new(&std::env::var("L1_OUTPUT").unwrap()).join("events.json"),
        )
        .unwrap(),
        &events,
    )
    .unwrap();
    let mut config = AppConfig::default();
    config.temporal_meter_t2 = std::env::var_os("L1_CONTINUOUS").is_some();
    let mut opts = options();
    opts.deterministic_analysis = false;
    opts.deterministic_footprints = false;
    let wiring = wire_runtime(
        &config,
        48_000,
        path,
        scenario,
        Arc::new(AtomicBool::new(false)),
        opts,
    )
    .unwrap();
    join(wiring).unwrap();
}

#[test]
#[ignore = "Flora closed-loop acquisition requires FLORA_SCENE and FLORA_REPORT"]
fn flora_closed_loop_acquisition() {
    let path = std::env::var("FLORA_SCENE").expect("Flora scene path");
    let report = std::env::var("FLORA_REPORT").expect("per-hop Flora report path");
    let scenario = compile_scenario_from_script(std::path::Path::new(&path), Some(21)).unwrap();
    use std::io::Write;
    let mut source_report = std::io::BufWriter::new(
        std::fs::File::create(std::path::Path::new(&report).with_extension("sources.jsonl"))
            .unwrap(),
    );
    let mut reporter = JsonlReporter::create(&report).unwrap();
    let mut opts = options();
    opts.offline_body_probe = Some(Box::new(move |state, tick, _, pcm| {
        assert!(pcm.into_iter().flatten().all(|x| x.is_finite()));
        let mut flora_pcm = [0.0_f32; 512];
        for source in &state.schedule_renderer.loo_pcm {
            if source.kind == crate::core::stream::loo::SourceKind::Flora {
                for (sum, value) in flora_pcm.iter_mut().zip(&source.habitat) {
                    *sum += value;
                }
            }
        }
        writeln!(
            source_report,
            "{}",
            serde_json::json!({
                "tick": tick,
                "only_flora_pcm": pcm[0] == flora_pcm,
                "flora_nonzero": flora_pcm.iter().any(|&x| x != 0.0),
            })
        )
        .unwrap();
        source_report.flush().unwrap();
        for flora in &state.pop.flora {
            let snapshot = flora.report(tick as f64 / 48_000.0, &state.current_landscape);
            let row = serde_json::to_value(&snapshot).unwrap();
            for key in [
                "biomass_mass_scan",
                "resource_state_scan",
                "biomass_consumption_scan",
                "external_consumption_scan",
            ] {
                assert!(
                    row[key]
                        .as_array()
                        .unwrap()
                        .iter()
                        .all(|x| x.as_f64().is_some_and(|x| x.is_finite() && x >= 0.0)),
                    "{key} at {tick}"
                );
            }
            let temporal = &row["temporal"];
            if temporal["active"] == true {
                assert_eq!(temporal["observation"]["observed_end"], tick);
                for event in temporal["events"].as_array().unwrap() {
                    if event["action"] == "born" {
                        assert_eq!(event["scheduled_tick"], event["tick"]);
                    }
                }
            }
            reporter
                .write_flora(flora, tick as f64 / 48_000.0, &state.current_landscape)
                .unwrap();
        }
        reporter.flush().unwrap();
    }));
    let config = AppConfig::default();
    let wiring = wire_runtime(
        &config,
        48_000,
        path,
        scenario,
        Arc::new(AtomicBool::new(false)),
        opts,
    )
    .unwrap();
    join(wiring).unwrap();
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
fn flora_meter_is_reserved_for_any_flora_and_observes_before_placement() {
    for continuous in [false, true] {
        for placement in ["", "let g = place(flora(), consonance(220, 880));"] {
            let has_flora = !placement.is_empty();
            let dir = std::env::temp_dir().join(format!(
                "flora-meter-{}-{continuous}-{}",
                std::process::id(),
                placement.len()
            ));
            std::fs::create_dir_all(&dir).unwrap();
            let script = dir.join("scenario.rhai");
            std::fs::write(&script, format!("wait(0.05); {placement} wait(0.05);")).unwrap();
            let scenario = compile_scenario_from_script(&script, Some(21)).unwrap();
            let seen = Arc::new(std::sync::Mutex::new([false; 2]));
            let observed = Arc::clone(&seen);
            let mut opts = options();
            opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
                assert_eq!(state.flora_meter.is_some(), has_flora);
                if tick > 0 {
                    assert_eq!(state.flora_rhythm.observed_end, has_flora.then_some(tick));
                    assert_eq!(state.flora_rhythm.known, has_flora);
                    observed.lock().unwrap()[usize::from(tick >= 2400)] = true;
                }
                if tick < 2400 {
                    assert!(state.pop.flora.is_empty());
                } else {
                    assert_eq!(state.pop.has_placed_flora, !placement.is_empty());
                }
            }));
            let mut config = AppConfig::default();
            config.analysis.nfft = 2048;
            config.temporal_meter_t2 = continuous;
            let wiring = wire_runtime(
                &config,
                48_000,
                "Flora meter reservation".into(),
                scenario,
                Arc::new(AtomicBool::new(false)),
                opts,
            )
            .unwrap();
            join(wiring).unwrap();
            assert_eq!(*seen.lock().unwrap(), [true, true]);
            std::fs::remove_dir_all(dir).unwrap();
        }
    }
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
fn nsgt_wiring_retains_only_requested_bus_power() {
    for kind in ["none", "gui", "report", "tap", "flora"] {
        let directory =
            std::env::temp_dir().join(format!("conchordal-nsgt-{}-{kind}", std::process::id()));
        std::fs::create_dir_all(&directory).unwrap();
        let script = directory.join("input.rhai");
        let flora = if kind == "flora" {
            "place(flora(), consonance(220, 880));"
        } else {
            ""
        };
        std::fs::write(&script,format!("seed(21); let g=place(sine().amp(0.05).sustain().anchor(),at(220)); {flora} wait(0.12); release(g); wait(0.05);")).unwrap();
        let mut scenario = compile_scenario_from_script(&script, None).unwrap();
        if kind == "tap" {
            scenario.temporal_mode = crate::scenario::TemporalMode::Observe;
        }
        let mut config = AppConfig::default();
        config.analysis.nfft = 2048;
        config.dcc.coupling_strength = 0.0;
        let core = build_analysis_runtime_core(&config, 48000);
        let mut reference = core.nsgt;
        let mut previous = vec![0.0; core.landscape.space.n_bins()];
        let mixed_enabled = kind != "none";
        let listener_enabled = matches!(kind, "gui" | "report" | "tap");
        let seen = Arc::new(std::sync::Mutex::new([false; 2]));
        let observed = Arc::clone(&seen);
        let mut opts = options();
        opts.listener_forced = true;
        opts.offline_body_probe = Some(Box::new(move |state, _, _, pcm| {
            if state.last_analysis_frame == state.frame_idx.checked_sub(1) && state.frame_idx > 0 {
                if mixed_enabled {
                    assert_eq!(state.current_landscape.nsgt_power, previous);
                } else {
                    assert!(state.current_landscape.nsgt_power.iter().all(|&p| p == 0.0));
                }
            }
            let mut seen = observed.lock().unwrap();
            seen[0] |= state.current_landscape.nsgt_power.iter().any(|&p| p > 0.0);
            seen[1] |= state.listener_twin.spectral_history.is_some();
            previous.copy_from_slice(reference.process_hop(pcm[0]));
        }));
        let (ui_tx, ui_rx) = crossbeam_channel::unbounded();
        if kind == "gui" {
            opts.ui_tx = Some(ui_tx);
        }
        if kind == "report" {
            opts.reporter = Some(
                JsonlReporter::create(directory.join("report.jsonl").to_str().unwrap()).unwrap(),
            );
        }
        let wiring = wire_runtime(
            &config,
            48000,
            "NSGT consumer wiring".into(),
            scenario,
            Arc::new(AtomicBool::new(false)),
            opts,
        )
        .unwrap();
        join(wiring).unwrap();
        assert_eq!(
            *seen.lock().unwrap(),
            [mixed_enabled, listener_enabled],
            "{kind}"
        );
        if kind == "gui" {
            let frames: Vec<_> = ui_rx.try_iter().collect();
            assert!(frames.iter().any(|f| f.spec.amps.iter().any(|&p| p > 0.0)));
            for frame in frames {
                for (amp, power) in frame.spec.amps.iter().zip(&frame.landscape.nsgt_power) {
                    assert_eq!(amp.to_bits(), power.sqrt().to_bits());
                }
            }
        }
        std::fs::remove_dir_all(directory).unwrap();
    }
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

#[test]
fn flora_force_stop_ends_generation_without_waiting_for_release() {
    for prototype in [false, true] {
        let dir = std::env::temp_dir().join(format!(
            "conchordal-flora-stop-{}-{prototype}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&dir).unwrap();
        let script = dir.join("input.rhai");
        std::fs::write(
            &script,
            "seed(21); place(flora().amp(0.2).fade(60), random(220, 880)); wait(0.01);",
        )
        .unwrap();
        let scenario = compile_scenario_from_script(&script, None).unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let trigger = stop.clone();
        let mut opts = options();
        opts.offline_body_probe = Some(Box::new(move |state, _, _, _| {
            if state.frame_idx == 8 {
                assert!(!state.pop.flora.is_empty());
                assert!(state.scenario_end_tick.is_some());
                trigger.store(true, Ordering::SeqCst);
            }
        }));
        let profile = dir.join("profile.json");
        opts.profile = Some(
            RunProfile::create(
                profile.to_str().unwrap(),
                21,
                false,
                0.,
                false,
                48000,
                512,
                None,
            )
            .unwrap(),
        );
        let mut config = AppConfig::default();
        config.analysis.nfft = 2048;
        config.render_prototype = prototype;
        let wiring = wire_runtime(
            &config,
            48000,
            "Flora stop".into(),
            scenario,
            stop.clone(),
            opts,
        )
        .unwrap();
        join(wiring).unwrap();
        assert!(stop.load(Ordering::SeqCst));
        let profile: serde_json::Value =
            serde_json::from_slice(&std::fs::read(profile).unwrap()).unwrap();
        let hops = profile["hops"].as_array().unwrap();
        assert_eq!(hops.len(), 9);
        #[cfg(feature = "profile-alloc")]
        {
            let counts: Vec<_> = hops
                .iter()
                .map(|hop| hop["worker_allocations"]["count"].as_u64().unwrap())
                .collect();
            println!("F12_WORKER_ALLOCATIONS prototype={prototype} counts={counts:?}");
        }
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn auditory_listener_clocks_and_device_rate_validation() {
    let script = std::env::temp_dir().join(format!(
        "conchordal-auditory-clock-{}.rhai",
        std::process::id()
    ));
    std::fs::write(&script, "seed(29); let v = place(sine().brain(\"drone\").sustain().anchor().amp(0.05), at(220)); wait(0.2); release(v); wait(0.02);").unwrap();
    let scenario = compile_scenario_from_script(&script, None).unwrap();
    let mut config = AppConfig::default();
    config.analysis.nfft = 2048;
    for (configured, actual) in [(48000, 16000), (16000, 16000)] {
        let mut invalid = config.clone();
        invalid.audio.sample_rate = configured;
        let error = wire_runtime(
            &invalid,
            actual,
            "invalid auditory rate".into(),
            scenario.clone(),
            Arc::new(AtomicBool::new(false)),
            options(),
        )
        .err()
        .expect("reject unsupported actual rate before constructing analysis");
        assert!(
            error.contains("audio.sample_rate") && error.contains("Nyquist"),
            "{error}"
        );
    }
    for deterministic in [true, false] {
        let rows = Arc::new(std::sync::Mutex::new(Vec::new()));
        let captured = rows.clone();
        let mut opts = options();
        opts.listener_forced = true;
        opts.deterministic_analysis = deterministic;
        opts.offline_body_probe = Some(Box::new(move |state, _, _, _| {
            if let Some(listener) = state.latest_listener_state {
                assert!(listener.generated_frame_id <= state.frame_idx);
                assert!(listener.analysis_frame_id < listener.generated_frame_id);
                assert_eq!(
                    listener.analysis_lag_frames,
                    listener.generated_frame_id - listener.analysis_frame_id
                );
                let expected_time = (listener.analysis_frame_id + 1) as f32 * 512.0 / 48000.0;
                assert!((listener.time_sec - expected_time).abs() < 1e-6);
                if deterministic {
                    assert_eq!(listener.generated_frame_id, state.frame_idx);
                    assert_eq!(listener.analysis_frame_id + 1, state.frame_idx);
                }
                captured.lock().unwrap().push((
                    state.frame_idx,
                    listener.generated_frame_id,
                    listener.analysis_frame_id,
                ));
            }
            // Give the asynchronous listener opportunities to publish without
            // imposing a wait in its real-time consumption path.
            if !deterministic {
                thread::sleep(Duration::from_millis(2));
            }
        }));
        let wiring = wire_runtime(
            &config,
            48000,
            "auditory listener clocks".into(),
            scenario.clone(),
            Arc::new(AtomicBool::new(false)),
            opts,
        )
        .unwrap();
        join_thread("worker", wiring.worker_handle)
            .unwrap()
            .unwrap();
        join_thread("analysis", wiring.analysis_handle).unwrap();
        join_thread("listener", wiring.listener_analysis_handle.unwrap()).unwrap();
        let rows = rows.lock().unwrap();
        assert!(
            rows.len() >= 2,
            "both modes must consume actual auditory analysis"
        );
        println!("AUD_M2_LISTENER_CLOCK deterministic={deterministic} rows={rows:?}");
    }
    std::fs::remove_file(script).unwrap();
}
