use super::*;

fn options() -> WiringOptions {
    WiringOptions {
        score: None,
        end_request: Arc::new(AtomicBool::new(false)),
        duration_sec: None,
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
        telemetry: false,
        audio_counters: None,
    }
}

#[test]
#[ignore = "TC1b-render lock; TC1B_CAPTURE, TC1B_SCENE and TC1B_OUTPUT required"]
fn tc1b_capture() {
    use std::io::Write;
    assert!(std::env::var_os("TC1B_CAPTURE").is_some());
    let path = std::env::var("TC1B_SCENE").unwrap();
    let directory = std::path::PathBuf::from(std::env::var("TC1B_OUTPUT").unwrap());
    std::fs::create_dir_all(&directory).unwrap();
    let config = AppConfig::default();
    crate::life::modal::register_modal();
    let (label, scenario, _, score) = super::scenario_load::load_scenario(
        Path::new(&path),
        Some(21),
        &config.audio,
        None,
        &AtomicBool::new(false),
        &AtomicBool::new(false),
    )
    .unwrap();
    // The bounded compatibility case retains the original 106-second prefix.
    let end = std::env::var("TC1B_END")
        .unwrap_or_else(|_| "130".into())
        .parse::<f32>()
        .unwrap();
    let mut pcm_file =
        std::io::BufWriter::new(std::fs::File::create(directory.join("habitat.bin")).unwrap());
    let mut decisions =
        std::io::BufWriter::new(std::fs::File::create(directory.join("decisions.jsonl")).unwrap());
    let stop = Arc::new(AtomicBool::new(false));
    let stop_at_end = stop.clone();
    let mut opts = options();
    opts.score = score;
    opts.listener_forced = true;
    opts.offline_body_probe = Some(Box::new(move |state, tick, _, pcm| {
        if tick >= (end as f64 * 48000.) as u64 {
            stop_at_end.store(true, Ordering::Relaxed);
        }
        pcm_file.write_all(&tick.to_le_bytes()).unwrap();
        let sources: Vec<_> = state
            .schedule_renderer
            .loo_pcm
            .iter()
            .filter(|s| s.kind == crate::core::stream::loo::SourceKind::Voice)
            .collect();
        pcm_file
            .write_all(&(sources.len() as u32).to_le_bytes())
            .unwrap();
        for x in pcm[0] {
            pcm_file.write_all(&x.to_le_bytes()).unwrap();
        }
        for source in sources {
            pcm_file.write_all(&source.id.to_le_bytes()).unwrap();
            pcm_file
                .write_all(&source.generation.to_le_bytes())
                .unwrap();
            for x in &source.habitat {
                pcm_file.write_all(&x.to_le_bytes()).unwrap();
            }
        }
        let rows = crate::core::tonal_context::acquisition::take(
            state.last_analysis_frame.map_or(0, |n| (n + 1) * 512),
        );
        for row in rows {
            writeln!(decisions, "{}", serde_json::to_string(&row).unwrap()).unwrap();
        }
        decisions.flush().unwrap();
    }));
    let wiring = wire_runtime(&config, 48000, label, scenario, stop, opts).unwrap();
    join(wiring).unwrap();
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
#[ignore = "F-19b PCM acquisition requires F19B_SCENE and F19B_PCM"]
fn flora_observer_pcm_acquisition() {
    use std::io::Write;
    let path = std::env::var("F19B_SCENE").unwrap();
    let scenario = compile_scenario_from_script(std::path::Path::new(&path), Some(21)).unwrap();
    let mut output =
        std::io::BufWriter::new(std::fs::File::create(std::env::var("F19B_PCM").unwrap()).unwrap());
    let mut opts = options();
    opts.offline_body_probe = Some(Box::new(move |state, _, _, pcm| {
        let mut sources = state
            .schedule_renderer
            .loo_pcm
            .iter()
            .filter(|s| s.kind == crate::core::stream::loo::SourceKind::Flora);
        let source = sources.next();
        assert!(sources.next().is_none(), "single Flora capture");
        output
            .write_all(&u32::from(source.is_some()).to_le_bytes())
            .unwrap();
        for &sample in pcm[0]
            .iter()
            .chain(source.map_or(&[0.0; 512][..], |s| &s.habitat))
        {
            output.write_all(&sample.to_le_bytes()).unwrap();
        }
    }));
    let wiring = wire_runtime(
        &AppConfig::default(),
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
    let (scenario, score, _) = ScriptHost::load_incremental(
        script.to_str().unwrap(),
        None,
        opts.duration_sec,
        &AtomicBool::new(false),
        &opts.end_request,
    )
    .unwrap();
    opts.score = score;
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
        None,
        Arc::new(AtomicBool::new(false)),
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
fn counted_audio_callback_error_during_playback_does_not_fail_the_worker() {
    let counters = Arc::new(crate::audio::output::AudioCallbackCounters::default());
    let counter = Arc::clone(&counters);
    let mut opts = options();
    opts.listener_forced = true;
    opts.audio_counters = Some(counters);
    opts.offline_body_probe = Some(Box::new(move |_, _, _, _| {
        counter.callback_errors_total.store(1, Ordering::Relaxed);
    }));
    let (wiring, _, dir) = fixture(opts, "seed(21); wait(0.2);");
    join(wiring).unwrap();
    assert!(
        std::fs::read_to_string(dir.join("run.jsonl"))
            .unwrap()
            .contains("rhythm_summary")
    );
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn stalled_audio_ring_pauses_and_resumes_the_worker_and_stop_still_works() {
    use ringbuf::traits::{Consumer, Split};
    let (producer, mut consumer) = ringbuf::HeapRb::<f32>::new(512).split();
    let counters = Arc::new(crate::audio::output::AudioCallbackCounters::default());
    counters.recovering.store(true, Ordering::Release);
    let hops = Arc::new(AtomicU64::new(0));
    let observed = Arc::clone(&hops);
    let mut opts = options();
    opts.audio_prod = Some(producer);
    opts.audio_counters = Some(counters);
    opts.offline_body_probe = Some(Box::new(move |_, _, _, _| {
        observed.fetch_add(1, Ordering::Relaxed);
    }));
    let (wiring, stop, dir) = fixture(opts, "seed(21); wait(10.0);");
    let deadline = Instant::now() + Duration::from_secs(10);
    while consumer.occupied_len() < 512 {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
    let first = hops.load(Ordering::Relaxed);
    assert_eq!(first, 1);
    thread::sleep(Duration::from_millis(20));
    assert_eq!(hops.load(Ordering::Relaxed), first);
    assert_eq!(consumer.skip(512), 512);
    while hops.load(Ordering::Relaxed) == first {
        assert!(Instant::now() < deadline);
        thread::sleep(Duration::from_millis(1));
    }
    assert!(!wiring.worker_handle.is_finished());
    stop.store(true, Ordering::SeqCst);
    while !wiring.worker_handle.is_finished() {
        assert!(
            Instant::now() < deadline,
            "stop must end a run blocked on ring vacancy"
        );
        thread::sleep(Duration::from_millis(1));
    }
    join(wiring).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn unsupported_recovery_after_finalization_fails_the_worker_result() {
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
    counters.failed.store(true, Ordering::Release);
    while !wiring.worker_handle.is_finished() && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(5));
    }
    let finished_without_user_exit = wiring.worker_handle.is_finished();
    stop.store(true, Ordering::SeqCst);
    let result = join(wiring).unwrap_err();
    assert!(finished_without_user_exit);
    assert!(
        result.contains("audio stream configuration unsupported during recovery"),
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
                let expected_time = (listener.analysis_frame_id + 1) as f64 * 512.0 / 48000.0;
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

#[test]
#[ignore = "PH-1 offline acquisition requires PH1_SCENE and PH1_OUTPUT"]
fn ph1_offline_acquisition() {
    use std::io::Write;
    let path = std::env::var("PH1_SCENE").unwrap();
    let directory = std::path::PathBuf::from(std::env::var("PH1_OUTPUT").unwrap());
    std::fs::create_dir_all(&directory).unwrap();
    let config = std::env::var("PH1_CONFIG").map_or_else(
        |_| AppConfig::default(),
        |path| AppConfig::load_or_default(Some(&path)).unwrap(),
    );
    crate::life::modal::register_modal();
    let (label, scenario, gain, score) = super::scenario_load::load_scenario(
        std::path::Path::new(&path),
        Some(21),
        &config.audio,
        None,
        &AtomicBool::new(false),
        &AtomicBool::new(false),
    )
    .unwrap();
    let mut reporter =
        JsonlReporter::create(directory.join("report.jsonl").to_str().unwrap()).unwrap();
    reporter.write_meta(scenario.seed).unwrap();
    if scenario.expression.is_none() {
        reporter
            .write_scene_markers(&scenario.scene_markers)
            .unwrap();
    }
    let mut evidence =
        std::io::BufWriter::new(std::fs::File::create(directory.join("commands.jsonl")).unwrap());
    let mut load = std::env::var_os("PH1_LOAD").map(|_| {
        std::io::BufWriter::new(std::fs::File::create(directory.join("load.jsonl")).unwrap())
    });
    let mut opts = options();
    opts.score = score;
    opts.reporter = Some(reporter);
    opts.listener_forced = true;
    if let Ok(path) = std::env::var("PH1_PROFILE") {
        opts.profile = Some(
            RunProfile::create(
                &path,
                scenario.seed,
                true,
                config.dcc.coupling_strength,
                true,
                48000,
                config.analysis.hop_size,
                None,
            )
            .unwrap(),
        );
    }
    let mut execution = std::env::var_os("HRY_EXECUTION").map(|_| {
        std::io::BufWriter::new(std::fs::File::create(directory.join("execution.jsonl")).unwrap())
    });
    // Profiling follows the ordinary reporter path without command-probe overhead.
    if opts.profile.is_none() {
        opts.offline_body_probe = Some(Box::new(move |state, tick, count, pcm| {
            if let (Some(writer), Some(trace)) = (
                &mut execution,
                &mut state.schedule_renderer.harmonic_execution,
            ) {
                for record in trace.drain(..) {
                    writeln!(writer, "{}", serde_json::to_string(&record).unwrap()).unwrap();
                }
            }
            if let Some(load) = load.as_mut() {
                writeln!(
                    load,
                    "{}",
                    serde_json::json!({
                        "tick": tick,
                        "voices_including_retiring": state.pop.voices.len(),
                        "tones": state.schedule_renderer.active_tone_count(),
                        "loo_sources": state.schedule_renderer.loo_pcm.len(),
                    })
                )
                .unwrap();
            }
            let batches: Vec<_> = state.phonation_batches_buf[..count]
                .iter()
                .map(|batch| {
                    serde_json::json!({
                        "source_id":batch.source_id,"commands":batch.cmds,"tones":batch.tones,
                    })
                })
                .collect();
            let sources:Vec<_>=state.schedule_renderer.loo_pcm.iter().map(|source|serde_json::json!({
            "id":source.id,"generation":source.generation,"kind":format!("{:?}",source.kind),
            "nonzero_samples":source.habitat.iter().filter(|&&x|x!=0.0).count(),
        })).collect();
            writeln!(
                evidence,
                "{}",
                serde_json::json!({"tick":tick,"batches":batches,"sources":sources,
                    "habitat_nonzero_samples":pcm[0].iter().filter(|&&x|x!=0.0).count(),
                    "presentation_nonzero_samples":pcm[1].iter().filter(|&&x|x!=0.0).count(),
                })
            )
            .unwrap();
        }));
    }
    let (tx, rx) = bounded::<Arc<[f32]>>(16);
    let guard = Some(Arc::new(LimiterMeter::default()));
    let wav = crate::audio::writer::WavOutput::run(
        rx,
        directory.join("audio.wav").to_string_lossy().into_owned(),
        48000,
        super::wiring::resolve_limiter_mode(&config),
        guard.clone(),
        gain,
    );
    opts.wav_tx = Some(tx);
    opts.guard_meter = guard;
    let wiring = wire_runtime(
        &config,
        48000,
        label,
        scenario,
        Arc::new(AtomicBool::new(false)),
        opts,
    )
    .unwrap();
    join(wiring).unwrap();
    wav.join().unwrap().unwrap();
}

#[test]
fn onset_habituation_wiring_starts_before_late_birth_and_zero_allocates_no_state() {
    use std::sync::Mutex;
    let root = std::env::temp_dir().join(format!(
        "conchordal-span-gap-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _directory = super::test_output_directory::TestOutputDirectory::new(root.clone());
    for weight in [0.0, 0.25] {
        let mut runs = Vec::new();
        for reporting in [false, true] {
            let source = format!(
                r#"
                let early=place(sine().sustain().amp(0.1),at(440.0));
                wait(1.0);
                let late=place(sine().sustain().metric().rest_on_habituation({weight}).amp(0.1),at(440.0));
                wait(1.0); release(late); release(early); wait(0.2);
            "#
            );
            let script = root.join(format!("{weight}-{reporting}.rhai"));
            std::fs::write(&script, source).unwrap();
            let scenario = compile_scenario_from_script(&script, Some(21)).unwrap();
            let capture = Arc::new(Mutex::new((false, false, Vec::new())));
            let copy = Arc::clone(&capture);
            let mut opts = options();
            if reporting {
                opts.reporter = Some(
                    JsonlReporter::create(root.join(format!("{weight}.jsonl")).to_str().unwrap())
                        .unwrap(),
                );
            }
            opts.offline_body_probe = Some(Box::new(move |state, tick, count, _| {
                let mut observed = copy.lock().unwrap();
                let frame = &state.current_landscape;
                if weight == 0.0 {
                    assert!(frame.onset_habituation_state_scan.is_empty());
                    assert!(
                        state
                            .pop
                            .voices
                            .iter()
                            .all(|v| v.phonation_engine.span_gap.is_none())
                    );
                } else if tick > 512 && tick < 48000 {
                    assert!(frame.onset_habituation_known);
                    assert_eq!(
                        frame.onset_habituation_state_scan.len(),
                        frame.space.n_bins()
                    );
                    observed.0 |= frame.onset_habituation_state_scan.iter().any(|&x| x > 0.0);
                }
                for voice in &state.pop.voices {
                    if let Some(gap) = &voice.phonation_engine.span_gap {
                        observed.1 = true;
                        assert!(gap.own >= 0.0 && gap.own <= 1.0);
                    }
                }
                for batch in &state.phonation_batches_buf[..count] {
                    observed.2.push(
                        serde_json::to_string(&(tick, batch.source_id, &batch.cmds, &batch.tones))
                            .unwrap(),
                    );
                }
            }));
            let wiring = wire_runtime(
                &AppConfig::default(),
                48000,
                script.to_string_lossy().into_owned(),
                scenario,
                Arc::new(AtomicBool::new(false)),
                opts,
            )
            .unwrap();
            join(wiring).unwrap();
            let value = capture.lock().unwrap();
            assert_eq!(value.0, weight > 0.0);
            assert_eq!(value.1, weight > 0.0);
            runs.push(value.2.clone());
        }
        assert_eq!(
            runs[0], runs[1],
            "reporting must not change actions or random draws"
        );
    }
}

#[test]
fn dwell_reporting_preserves_real_tone_commands_and_records_stopped_end_state() {
    use std::sync::Mutex;
    let root = std::env::temp_dir().join(format!(
        "conchordal-dwell-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _directory = super::test_output_directory::TestOutputDirectory::new(root.clone());
    for stopped in [false, true] {
        let mut runs = Vec::new();
        for reporting in [false, true] {
            let script = root.join(format!("{stopped}-{reporting}.rhai"));
            std::fs::write(&script, "place(sine().sustain().metric().seek_consonance().dwell(1).proposal_interval(0.1).pitch_apply_mode(\"rearticulate\").rearticulate_above(50).rest_on_habituation(0.25).leave_self_out(true),at(330)); wait(3.0);").unwrap();
            let scenario = compile_scenario_from_script(&script, Some(21)).unwrap();
            let report = root.join(format!("{stopped}.jsonl"));
            let capture = Arc::new(Mutex::new(Vec::new()));
            let copy = capture.clone();
            let mut opts = options();
            opts.listener_forced = true;
            if reporting {
                opts.reporter = Some(JsonlReporter::create(report.to_str().unwrap()).unwrap());
            }
            opts.offline_body_probe = Some(Box::new(move |state, tick, count, _| {
                for batch in &state.phonation_batches_buf[..count] {
                    copy.lock().unwrap().push(
                        serde_json::to_string(&(tick, batch.source_id, &batch.cmds, &batch.tones))
                            .unwrap(),
                    );
                }
            }));
            let mut config = AppConfig::default();
            if stopped {
                config.psychoacoustics.consonance.field.level.theta = -100.0;
                config.psychoacoustics.habituation.enabled = false;
            }
            let wiring = wire_runtime(
                &config,
                48000,
                script.to_string_lossy().into_owned(),
                scenario,
                Arc::new(AtomicBool::new(false)),
                opts,
            )
            .unwrap();
            join(wiring).unwrap();
            runs.push(capture.lock().unwrap().clone());
            if reporting {
                let records: Vec<serde_json::Value> = std::fs::read_to_string(report)
                    .unwrap()
                    .lines()
                    .map(|s| serde_json::from_str(s).unwrap())
                    .filter(|r: &serde_json::Value| r["type"] == "dwell")
                    .collect();
                assert_eq!(records.iter().filter(|r| r["event"] == "end").count(), 1);
                if stopped {
                    assert!(records.iter().all(|r| r["event"] != "proposal"));
                    assert_eq!(records.last().unwrap()["phase"], 0.0);
                    assert_eq!(records.last().unwrap()["level"], 1.0);
                } else {
                    assert!(records.iter().any(|r| r["event"] == "proposal"));
                }
            }
        }
        assert_eq!(runs[0], runs[1], "reporting must not change Tone commands");
    }
}

#[test]
fn dwell_proposals_are_reported_before_an_early_stop() {
    let root = std::env::temp_dir().join(format!(
        "conchordal-dwell-abort-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let _directory = super::test_output_directory::TestOutputDirectory::new(root.clone());
    let script = root.join("scene.rhai");
    std::fs::write(&script, "place(sine().sustain().seek_consonance().dwell(0.5).proposal_interval(0.001),at(330)); wait(3.0);").unwrap();
    let scenario = compile_scenario_from_script(&script, Some(21)).unwrap();
    let report = root.join("report.jsonl");
    let stop = Arc::new(AtomicBool::new(false));
    let signal = stop.clone();
    let mut opts = options();
    opts.reporter = Some(JsonlReporter::create(report.to_str().unwrap()).unwrap());
    opts.offline_body_probe = Some(Box::new(move |_, _, _, _| {
        signal.store(true, Ordering::Release);
    }));
    let wiring = wire_runtime(
        &AppConfig::default(),
        48000,
        script.to_string_lossy().into_owned(),
        scenario,
        stop,
        opts,
    )
    .unwrap();
    join(wiring).unwrap();
    let records: Vec<serde_json::Value> = std::fs::read_to_string(report)
        .unwrap()
        .lines()
        .map(|s| serde_json::from_str(s).unwrap())
        .filter(|r: &serde_json::Value| r["type"] == "dwell")
        .collect();
    assert!(records.iter().any(|r| r["event"] == "proposal"));
    assert_eq!(records.iter().filter(|r| r["event"] == "end").count(), 1);
}

#[test]
fn open_queue_exhaustion_keeps_voices_and_respawn_running() {
    for respawn in [false, true] {
        let reached = Arc::new(AtomicBool::new(false));
        let observed = reached.clone();
        let mut opts = options();
        opts.duration_sec = Some(0.15);
        opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
            if tick == 10 * 512 {
                assert!(!state.finish_begun);
                assert_eq!(state.playback_state, PlaybackState::Playing);
                assert!(!state.pop.voices.is_empty());
                if respawn {
                    let (births, deaths) = state.pop.take_lifecycle_counts();
                    assert!(births > 1 && deaths > 0, "births={births} deaths={deaths}");
                }
                observed.store(true, Ordering::SeqCst);
            }
        }));
        let source = format!(
            "seed(21); endless(); place(sine().sustain().anchor(){}, at(220));",
            if respawn {
                ".respawn_random().respawn_background_death_rate(10000)"
            } else {
                ""
            }
        );
        let (wiring, _, dir) = fixture(opts, &source);
        join(wiring).unwrap();
        assert!(reached.load(Ordering::SeqCst));
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn duration_dispatches_due_events_then_releases_and_keeps_old_fades() {
    let reached = Arc::new(AtomicBool::new(false));
    let observed = reached.clone();
    let mut opts = options();
    opts.duration_sec = Some(1024.0 / 48000.0);
    opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
        if tick == 1024 {
            assert_eq!(state.playback_state, PlaybackState::Ending);
            assert_eq!(
                state.pop.global_coupling, 0.25,
                "bound-hop event must run first"
            );
            let finish = state.end_sequence.as_ref().unwrap().finish_tick;
            assert!(finish >= 48000 / 5, "older 0.2s Flora fade must finish");
            assert!(state.pop.voices.iter().all(|v| v.release.remove_pending));
            assert!(state.pop.voices.iter().any(|v| v.release.sec == 0.1));
            assert!(state.pop.voices.iter().any(|v| v.release.sec == 0.02));
            observed.store(true, Ordering::SeqCst);
        }
        if let Some(end) = &state.end_sequence {
            if tick < end.finish_tick {
                assert!(!state.finish_begun);
                // A fading Voice must survive until its gain reaches zero.
                if tick < 4096 {
                    assert!(
                        state
                            .pop
                            .voices
                            .iter()
                            .any(|v| v.release.sec == 0.1 && v.release_gain() > 0.0)
                    );
                }
            } else {
                assert!(state.finish_begun);
                assert!(
                    state
                        .pop
                        .flora
                        .iter()
                        .all(|f| f.remaining_release_ticks() == 0)
                );
            }
        }
    }));
    let (wiring, _, dir) = fixture(
        opts,
        r#"
        seed(21); endless();
        let old = place(sine().anchor().sustain().adsr(0, 0, 1, 0.1), at(220));
        let moss = place(flora().fade(0.2), random(220, 880));
        wait(512.0 / 48000.0);
        release(old); release(moss); flush();
        wait(512.0 / 48000.0);
        place(sine().anchor().sustain().adsr(0, 0, 1, 0.02), at(330));
        set_global_coupling(0.25);
        wait(10); set_global_coupling(0.9);
    "#,
    );
    join(wiring).unwrap();
    assert!(reached.load(Ordering::SeqCst));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn human_end_discards_due_events_and_pending_scripted_finish() {
    let end_request = Arc::new(AtomicBool::new(false));
    let request = end_request.clone();
    let reached = Arc::new(AtomicBool::new(false));
    let observed = reached.clone();
    let mut opts = options();
    opts.end_request = end_request;
    opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
        if tick == 0 {
            request.store(true, Ordering::SeqCst);
        }
        if tick == 512 {
            assert_eq!(
                state.pop.global_coupling, 1.0,
                "human end discards undispatched events"
            );
            assert_eq!(state.playback_state, PlaybackState::Ending);
            assert!(!state.finish_begun);
            assert_eq!(
                state.conductor.current_scene_name(tick).as_deref(),
                Some("active")
            );
            observed.store(true, Ordering::SeqCst);
        }
    }));
    let (wiring, _, dir) = fixture(
        opts,
        "seed(21); section(\"active\", || {}); place(sine().anchor().sustain().adsr(0,0,1,0.05), at(220)); wait(512.0/48000.0); section(\"cancelled\", || {}); set_global_coupling(0.25); wait(100);",
    );
    join(wiring).unwrap();
    assert!(reached.load(Ordering::SeqCst));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn end_during_recovery_exits_even_when_ring_is_full() {
    use ringbuf::traits::{Producer, Split};
    for during_fade in [false, true] {
        let (mut producer, _consumer) = ringbuf::HeapRb::<f32>::new(512).split();
        if !during_fade {
            assert_eq!(producer.push_slice(&[0.0; 512]), 512);
        }
        let end_request = Arc::new(AtomicBool::new(false));
        let counters = Arc::new(crate::audio::output::AudioCallbackCounters::default());
        let request = end_request.clone();
        let recovering = counters.clone();
        let mut opts = options();
        opts.end_request = end_request.clone();
        opts.audio_prod = Some(producer);
        opts.audio_counters = Some(counters.clone());
        if during_fade {
            opts.duration_sec = Some(0.0);
            opts.offline_body_probe = Some(Box::new(move |state, _, _, _| {
                assert!(state.end_sequence.is_some());
                assert!(!state.finish_begun);
                recovering.recovering.store(true, Ordering::Release);
            }));
        } else {
            counters.recovering.store(true, Ordering::Release);
        }
        let (wiring, stop, dir) = fixture(
            opts,
            "seed(21); endless(); place(sine().sustain().adsr(0,0,1,10), at(220));",
        );
        if !during_fade {
            request.store(true, Ordering::SeqCst);
        }
        let deadline = Instant::now() + Duration::from_secs(5);
        while !wiring.worker_handle.is_finished() && Instant::now() < deadline {
            thread::sleep(Duration::from_millis(1));
        }
        let completed = wiring.worker_handle.is_finished();
        stop.store(true, Ordering::SeqCst);
        join(wiring).unwrap();
        assert!(
            completed,
            "end request must not wait for ring vacancy during recovery"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
fn open_harmonic_queue_does_not_finish_on_exhaustion() {
    let reached = Arc::new(AtomicBool::new(false));
    let observed = reached.clone();
    let mut opts = options();
    opts.duration_sec = Some(0.15);
    opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
        if tick == 10 * 512 {
            assert!(!state.finish_begun);
            assert!(
                state
                    .pop
                    .voices
                    .iter()
                    .any(|v| v.harmonic_member && !v.release.remove_pending)
            );
            observed.store(true, Ordering::SeqCst);
        }
    }));
    let (wiring, _, dir) = fixture(
        opts,
        "endless(); harmonic_rhythm(1); place(sine().sustain().seek_consonance().pitch_apply_mode(\"rearticulate\").rearticulate_above(0), at(220));",
    );
    join(wiring).unwrap();
    assert!(reached.load(Ordering::SeqCst));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn duration_keeps_existing_rain_release_tail() {
    let reached = Arc::new(AtomicBool::new(false));
    let observed = reached.clone();
    let mut opts = options();
    opts.duration_sec = Some(0.1);
    opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
        if let Some(end) = &state.end_sequence {
            assert!(end.finish_tick >= 21_600);
            if tick < 20_000 {
                assert!(!state.finish_begun);
                assert!(!state.pop.rain.is_empty());
                observed.store(true, Ordering::SeqCst);
            }
        }
    }));
    let (wiring, _, dir) = fixture(
        opts,
        "endless(); let r=place(rain().fade(0.4),random(300,6000)); wait(0.05); release(r);",
    );
    join(wiring).unwrap();
    assert!(reached.load(Ordering::SeqCst));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn incremental_cycle_state_is_bounded_with_respawn_and_flora_v2() {
    let checkpoints = Arc::new(std::sync::Mutex::new(Vec::new()));
    let observed = checkpoints.clone();
    let mut opts = options();
    let end = opts.end_request.clone();
    opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
        if [193 * 512, 289 * 512].contains(&tick) {
            let stream = &state.conductor.score.as_ref().unwrap().stream;
            let score = stream.state.lock().unwrap();
            let sizes = (
                state.pop.population_count(),
                state.loo_populations.len(),
                score.ring.len(),
                score.script_sizes,
                score.warnings.len(),
            );
            eprintln!("V2 tick={tick} sizes={sizes:?}");
            observed.lock().unwrap().push(sizes);
            if tick == 289 * 512 {
                let (births, deaths) = state.pop.take_lifecycle_counts();
                assert!(
                    births > 5 && deaths > 0,
                    "churn must actually occur: {births}/{deaths}"
                );
                end.store(true, Ordering::SeqCst);
            }
        }
    }));
    let (wiring, _, dir) = fixture(
        opts,
        r#"
        seed(21); endless();
        let base = place(sine().sustain().anchor().leave_self_out(true)
            .respawn_random().respawn_background_death_rate(5), at(220));
        loop {
            section("cycle", || {
                let p = place(sine().sustain().anchor().leave_self_out(true).adsr(0,0,1,0.01), at(330));
                let f = place(flora().fade(0.01), random(110,880));
                wait(0.256); release(p); release(f); flush(); release(p);
            });
            wait(0.768);
        }
    "#,
    );
    join(wiring).unwrap();
    let checkpoints = checkpoints.lock().unwrap();
    assert_eq!(checkpoints.len(), 2);
    assert_eq!(checkpoints[0], checkpoints[1]);
    assert_eq!(checkpoints[0].3, (1, 0, 0));
    assert_eq!(checkpoints[0].4, 1);
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn incremental_duration_reports_and_presynth_are_repeatable_v4() {
    let mut results = Vec::new();
    for _ in 0..2 {
        let snapshots = Arc::new(std::sync::Mutex::new(Vec::new()));
        let saved = snapshots.clone();
        let mut opts = options();
        opts.duration_sec = Some(0.128);
        opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
            saved.lock().unwrap().push((
                tick,
                state.finish_begun,
                state
                    .pop
                    .voices
                    .iter()
                    .map(|voice| {
                        (
                            voice.id(),
                            voice.metadata.population_id,
                            voice.release.gain.to_bits(),
                            voice.target_pitch_log2().to_bits(),
                        )
                    })
                    .collect::<Vec<_>>(),
            ));
            if tick == 6144 {
                assert_eq!(state.playback_state, PlaybackState::Ending);
                assert!(state.pop.voices.iter().all(|v| v.release.remove_pending));
                assert_eq!(state.conductor.current_scene.as_deref(), Some("bounded"));
            }
        }));
        let (wiring, _, dir) = fixture(
            opts,
            r#"
            seed(21); endless();
            section("bounded", || {
                let p = place(sine().anchor().sustain().adsr(0,0,1,0.01), at(220));
                loop { p.amp(0.1); wait(0.02); }
            });
        "#,
        );
        join(wiring).unwrap();
        let report = std::fs::read_to_string(dir.join("run.jsonl")).unwrap();
        let report: Vec<serde_json::Value> = report
            .lines()
            .map(|line| {
                let mut row: serde_json::Value = serde_json::from_str(line).unwrap();
                if row["type"] == "hop_timing" {
                    for key in ["elapsed_us", "analysis_wait_us", "listener_wait_us"] {
                        row.as_object_mut().unwrap().remove(key);
                    }
                }
                row
            })
            .collect();
        results.push((snapshots.lock().unwrap().clone(), report));
        std::fs::remove_dir_all(dir).unwrap();
    }
    assert!(!results[0].0.is_empty());
    assert!(!results[0].1.is_empty());
    assert_eq!(results[0].0, results[1].0);
    assert_eq!(results[0].1.len(), results[1].1.len());
    for (a, b) in results[0].1.iter().zip(&results[1].1) {
        assert_eq!(a, b);
    }
}

#[test]
fn endless_onset_history_starts_before_future_consumers() {
    let reached = Arc::new(AtomicBool::new(false));
    let observed = reached.clone();
    let mut opts = options();
    opts.duration_sec = Some(1.1);
    opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
        if tick == 20 * 512 {
            assert!(state.current_landscape.onset_habituation_known);
            assert!(
                !state
                    .current_landscape
                    .onset_habituation_state_scan
                    .is_empty()
            );
            assert!(
                state
                    .pop
                    .voices
                    .iter()
                    .all(|v| v.phonation_engine.span_gap.is_none())
            );
            observed.store(true, Ordering::SeqCst);
        }
    }));
    let (wiring, _, dir) = fixture(
        opts,
        "endless(); place(sine().sustain(),at(440)); wait(1); place(sine().sustain().metric().rest_on_habituation(0.25),at(330)); wait(1);",
    );
    join(wiring).unwrap();
    assert!(reached.load(Ordering::SeqCst));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn numerical_reference_on_bundled_sample_histories() {
    use crate::core::history_prediction::reference;
    for sample in ["01_a_single_voice", "08_murmuration", "09_rain"] {
        let scenario =
            compile_scenario_from_script(Path::new(&format!("samples/{sample}.rhai")), Some(21))
                .unwrap();
        let counts = Arc::new(std::sync::Mutex::new(reference::Counts::default()));
        let captured = counts.clone();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_probe = stop.clone();
        let mut opts = options();
        let mut started = false;
        opts.offline_body_probe = Some(Box::new(move |_, tick, _, _| {
            if !started {
                reference::start();
                started = true;
            }
            *captured.lock().unwrap() = reference::counts();
            if tick >= 8 * 48000 {
                stop_probe.store(true, Ordering::Relaxed);
            }
        }));
        let config = AppConfig::default();
        let wiring = wire_runtime(&config, 48000, sample.into(), scenario, stop, opts).unwrap();
        join(wiring).unwrap();
        let result = *counts.lock().unwrap();
        println!("numerical sample={sample} {result:?}");
        // Sustained-only pieces need not issue comparison candidates.
        assert!(result.passed[2] > 0, "{result:?}");
        assert_eq!(result.failed, [0; 3], "{result:?}");
        assert_eq!(result.inconclusive, [0; 3], "{result:?}");
        if sample != "01_a_single_voice" {
            result.assert_passed();
        }
    }
}

#[test]
fn endless_harmonic_history_starts_before_future_consumers() {
    for temporal in [false, true] {
        let reached = Arc::new(AtomicBool::new(false));
        let observed = reached.clone();
        let mut opts = options();
        opts.duration_sec = Some(0.6);
        opts.offline_body_probe = Some(Box::new(move |state, tick, _, _| {
            if tick == 20 * 512 {
                let support = state
                    .current_landscape
                    .support
                    .as_ref()
                    .expect("reserved support scan");
                assert!(support.end_sample > 0);
                assert!(!state.current_landscape.support_weights.active());
                assert_eq!(state.current_landscape.harmonic_relations_weight, 0.);
                assert_eq!(state.harmonic_relations.is_some(), temporal);
                if let Some(relations) = &state.harmonic_relations {
                    assert_eq!(relations.sources, 36);
                    assert!(relations.observations.load(Ordering::Relaxed) > 0);
                }
                observed.store(true, Ordering::SeqCst);
            }
        }));
        let declaration = if temporal {
            "observe_harmonic_relations(36);"
        } else {
            ""
        };
        let control = if temporal {
            "follow_harmonic_relations(1);"
        } else {
            "set_support_focus_preference(1);"
        };
        let source = format!(
            "endless(); {declaration} place(sine().sustain(),at(440)); wait(0.5); {control} wait(1);"
        );
        let (wiring, _, dir) = fixture(opts, &source);
        join(wiring).unwrap();
        assert!(reached.load(Ordering::SeqCst));
        std::fs::remove_dir_all(dir).unwrap();
    }
}

#[test]
#[cfg(feature = "profile-alloc")]
#[ignore = "HV-INT observer cost acquisition requires the timing lock and HVINT_COST_OUTPUT"]
fn harmonic_observer_reservation_cost() {
    use crate::core::stream::loo::SourceKind;
    use std::collections::BTreeMap;
    type SourceKey = (u8, u64, u32);
    type Input = (Vec<f32>, Vec<(SourceKey, Vec<f32>)>);
    let inputs = Arc::new(std::sync::Mutex::new(Vec::<Input>::new()));
    let captured = inputs.clone();
    let mut opts = options();
    opts.offline_body_probe = Some(Box::new(move |state, _, _, pcm| {
        let own = state
            .schedule_renderer
            .loo_pcm
            .iter()
            .filter(|s| s.analyze_hr)
            .map(|s| {
                (
                    (
                        if s.kind == SourceKind::Voice { 0 } else { 1 },
                        s.id,
                        s.generation,
                    ),
                    s.habitat.clone(),
                )
            })
            .collect();
        captured.lock().unwrap().push((pcm[0].to_vec(), own));
    }));
    let source = std::fs::read_to_string("tests/research/harmonic_relations_load.rhai")
        .unwrap()
        .replace(
            "follow_harmonic_relations(0.8)",
            "follow_harmonic_relations(0)",
        );
    let (wiring, _, dir) = fixture(opts, &source);
    join(wiring).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    let inputs = Arc::try_unwrap(inputs).unwrap().into_inner().unwrap();
    let max_own = inputs.iter().map(|(_, s)| s.len()).max().unwrap();
    assert_eq!(max_own, 4, "two Voice and two Flora H/R observers");
    let mut results = Vec::new();
    for support in [false, true, true, false] {
        let initial_heap = crate::runtime_profile::heap_live_bytes();
        let core = build_analysis_runtime_core(&AppConfig::default(), 48000);
        let mut mixed = AnalysisStream::new(core.lparams.clone(), core.nsgt).with_nsgt_power(true);
        if support {
            let geometry = crate::core::harmonic_support::Geometry::new(
                &mixed.last().space,
                &core.lparams.harmonicity_kernel,
            );
            mixed.enable_support(geometry, true);
        }
        let mut own = BTreeMap::<SourceKey, AnalysisStream>::new();
        let mut scratch = vec![0.; 512];
        let mut times = Vec::with_capacity(inputs.len());
        let mut heap_peak = 0;
        for (pcm, sources) in &inputs {
            own.retain(|key, _| sources.iter().any(|(k, _)| key == k));
            for (key, _) in sources {
                own.entry(*key).or_insert_with(|| mixed.voice_observer());
            }
            let start = Instant::now();
            std::hint::black_box(mixed.process(pcm));
            for (key, self_pcm) in sources {
                for ((dst, mix), source) in scratch.iter_mut().zip(pcm).zip(self_pcm) {
                    *dst = mix - source;
                }
                std::hint::black_box(own.get_mut(key).unwrap().process(&scratch));
            }
            times.push(start.elapsed().as_secs_f64() * 1000.);
            heap_peak = heap_peak
                .max(crate::runtime_profile::heap_live_bytes().saturating_sub(initial_heap));
        }
        results.push(
            serde_json::json!({"support":support,"heap_peak_bytes":heap_peak,"hop_ms":times}),
        );
    }
    let output = std::env::var("HVINT_COST_OUTPUT").unwrap();
    std::fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({
        "sample_rate":48000,"hop":512,"hops":inputs.len(),"maximum_self_observers":max_own,
        "source":"tests/research/harmonic_relations_load.rhai; temporal weight zero; actual habitat and own PCM",
        "scope":"auditory mixed plus actual Voice/Flora self-subtracted H/R streams; excludes publication, resource-only Flora FFT, consumer and background work",
        "windows":results
    })).unwrap()).unwrap();
}

#[test]
fn listener_prediction_is_invariant_to_hidden_habitat_and_hidden_id_renaming() {
    let script = std::env::temp_dir().join(format!(
        "conchordal-gui1-invariance-{}.rhai",
        std::process::id()
    ));
    let mut traces = Vec::new();
    let mut pcm_traces = Vec::new();
    let mut id_traces = Vec::new();
    for (hidden_hz, reserved, enabled) in [
        (110., 0, true),
        (440., 0, true),
        (110., 1 << 20, true),
        (110., 0, false),
    ] {
        std::fs::write(&script, format!(r#"
seed(21);
let hidden = harmonic().send(habitat_bus).sustain().anchor().amp(0.03);
place(hidden, at({hidden_hz}));
let audible = harmonic().brain("drone").send(presentation_bus).sustain().anchor().amp(0.06).adsr(0.005, 0.02, 0.8, 0.02);
for i in 0..16 {{
    let v = place(audible, at(if i % 2 == 0 {{ 220.0 }} else {{ 330.0 }}));
    wait(0.12); release(v); wait(0.38);
}}
"#)).unwrap();
        let mut scenario = compile_scenario_from_script(&script, Some(21)).unwrap();
        for event in &mut scenario.events {
            for action in &mut event.actions {
                if let crate::scenario::Action::Spawn {
                    ids: crate::scenario::SpawnIds::Assigned(ids),
                    ..
                } = action
                {
                    for id in ids {
                        // Rename the habitat-only source; presentation PCM stays fixed.
                        if *id == 1 {
                            *id ^= reserved;
                        }
                    }
                }
            }
        }
        let mut opts = options();
        opts.listener_forced = true;
        opts.reserve_runtime_ids_through = reserved;
        let rows = Arc::new(std::sync::Mutex::new(Vec::new()));
        let pcm = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ids = Arc::new(std::sync::Mutex::new(Vec::new()));
        let (capture, audio, identity) = (rows.clone(), pcm.clone(), ids.clone());
        opts.offline_body_probe = Some(Box::new(move |state, _, _, buses| {
            state.listener_twin.set_prediction_enabled(enabled);
            if let Some(listener) = state.latest_listener_state {
                capture.lock().unwrap().push(listener.prediction);
            }
            audio
                .lock()
                .unwrap()
                .extend(buses[1].iter().map(|x| x.to_bits()));
            identity
                .lock()
                .unwrap()
                .extend(state.pop.voices.iter().map(|v| v.id));
        }));
        let wiring = wire_runtime(
            &AppConfig::default(),
            48000,
            "GUI-1 invariance".into(),
            scenario,
            Arc::new(AtomicBool::new(false)),
            opts,
        )
        .unwrap();
        join(wiring).unwrap();
        traces.push(rows.lock().unwrap().clone());
        pcm_traces.push(pcm.lock().unwrap().clone());
        id_traces.push(ids.lock().unwrap().clone());
    }
    std::fs::remove_file(script).unwrap();
    assert!(
        traces[0]
            .iter()
            .any(|p| p.completed.is_some_and(|d| d.forecast.q.is_some())),
        "invariance must exercise supported predictions"
    );
    assert!(traces[0] == traces[1], "habitat-only intervention");
    assert!(traces[0] == traces[2], "hidden renderer ID renaming");
    assert!(
        id_traces[0] != id_traces[2],
        "the ID intervention must actually rename Voices"
    );
    assert!(traces[3].iter().all(|p| p.forecast.is_none()));
    for other in &pcm_traces[1..] {
        assert!(
            pcm_traces[0] == *other,
            "fixed presentation / passive predictor"
        );
    }
}

#[test]
#[ignore = "GUI-1 ABBA replay: timing lock, GUI1_WAV and GUI1_COST_OUTPUT required"]
fn gui1_listener_replay_cost() {
    use std::io::Write;
    let path = std::env::var("GUI1_WAV").unwrap();
    let enabled = std::env::var("GUI1_E0").unwrap() == "1";
    let output = std::env::var("GUI1_COST_OUTPUT").unwrap();
    let config = AppConfig::default();
    let core = build_analysis_runtime_core(&config, 48000);
    let mut stream = AnalysisStream::new(core.lparams.clone(), core.nsgt).with_nsgt_power(true);
    let mut twin = ListenerTwin::with_sample_rate(
        48000.,
        crate::listener_twin::ListenerTwinConfig {
            prediction_enabled: enabled,
            ..Default::default()
        },
    );
    let mut hab = crate::core::habituation::HabituationField::new(
        &core.lparams.habituation,
        core.lparams.consonance_representation.theta,
        core.landscape.space.n_bins(),
    );
    let mut reader = hound::WavReader::open(path).unwrap();
    assert_eq!(reader.spec().sample_rate, 48000);
    assert_eq!(reader.spec().bits_per_sample, 16);
    let channels = reader.spec().channels as usize;
    let capacity = reader.duration() as usize / 512;
    let mut samples = reader.samples::<i16>();
    let mut hop = [0.0_f32; 512];
    let mut rows = Vec::with_capacity(capacity);
    let mut frame = 0_u64;
    loop {
        let mut complete = true;
        for x in &mut hop {
            let Some(value) = samples.next() else {
                complete = false;
                break;
            };
            *x = f32::from(value.unwrap()) / 32768.;
            for _ in 1..channels {
                samples.next().unwrap().unwrap();
            }
        }
        if !complete {
            break;
        }
        let start = Instant::now();
        let mut landscape = stream.process(&hop).clone();
        let analysis = start.elapsed().as_secs_f64() * 1e6;
        let start = Instant::now();
        landscape.recompute_consonance(&core.lparams);
        analysis_sync::drive_and_apply_habituation(
            &mut landscape,
            &mut hab,
            &core.lparams,
            512. / 48000.,
        );
        let habituation = start.elapsed().as_secs_f64() * 1e6;
        let start = Instant::now();
        crate::runtime_profile::begin_allocations();
        twin.observe_presentation_audio(frame * 512, frame, &hop);
        let state = twin.observe_presentation_landscape(
            (frame + 1) as f64 * 512. / 48000.,
            frame + 1,
            frame,
            &landscape,
        );
        let allocations = crate::runtime_profile::finish_allocations();
        let listener = start.elapsed().as_secs_f64() * 1e6;
        rows.push((
            frame,
            analysis,
            habituation,
            listener,
            allocations,
            state.prediction.forecast.map_or(0, |f| f.examples),
        ));
        frame += 1;
    }
    let mut out = std::io::BufWriter::new(std::fs::File::create(output).unwrap());
    for (frame, analysis, habituation, listener, allocations, examples) in rows {
        writeln!(out, "{}", serde_json::json!({"frame":frame, "analysis_us":analysis, "habituation_us":habituation, "listener_us":listener, "allocations":allocations, "examples":examples})).unwrap();
    }
}
