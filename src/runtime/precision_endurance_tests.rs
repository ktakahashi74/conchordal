use super::*;
use crate::core::history_prediction::reference;
use std::io::Write;

#[test]
#[ignore = "V7 long-run precision acquisition; release, owned memory limit, no audio device"]
fn indefinite_estimator_precision_v7() {
    let debug_build = cfg!(debug_assertions);
    assert!(!debug_build, "run V7 with --release");
    let output = std::env::var("IP_V7_OUTPUT").expect("IP_V7_OUTPUT checkpoint JSONL path");
    let snapshots = std::env::var("IP_V7_SNAPSHOTS").expect("IP_V7_SNAPSHOTS directory");
    std::fs::create_dir(&snapshots).expect("use a new snapshot directory for each acquisition");
    let mut journal = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)
        .unwrap();
    let checkpoints: Vec<u64> = std::env::var("IP_V7_CHECKPOINTS")
        .unwrap_or_else(|_| "1000000,10000000,30000000".into())
        .split(',')
        .map(|s| s.parse().unwrap())
        .collect();
    assert!(checkpoints[0] > 0 && checkpoints.windows(2).all(|w| w[0] < w[1]));
    let full = checkpoints == [1_000_000, 10_000_000, 30_000_000];
    let script = Path::new("tests/fixtures/indefinite_precision.rhai");
    let scenario = compile_scenario_from_script(script, Some(21)).unwrap();
    assert!(matches!(scenario.end, crate::scenario::ScenarioEnd::Open));
    let stop = Arc::new(AtomicBool::new(false));
    let stop_probe = stop.clone();
    let result = Arc::new(std::sync::Mutex::new(None));
    let captured = result.clone();
    let mut identities = None;
    let mut next = 0;
    let mut stop_after = *checkpoints.last().unwrap();
    let mut previous = reference::Counts::default();
    let mut churn_generation = 0;
    let mut last_churn_id = None;
    let mut churn_births = 0_u64;
    let start = Instant::now();
    let mut heartbeat = Instant::now();
    let probe: OfflineBodyProbe = Box::new(move |state, tick, _, _| {
        let observer = state.temporal_expectation.as_mut().unwrap();
        if identities.is_none() {
            assert_eq!(tick, 0);
            let drone = &state.pop.voices[0];
            let drone_id = drone.id();
            let generation = drone.metadata.generation;
            let own = state
                .schedule_renderer
                .self_sound_history(drone_id)
                .unwrap();
            let (shared_id, shared_issues) = observer.reference_identity();
            let (own_id, own_issues) = own.external.reference_identity();
            assert_eq!((shared_issues, own_issues), (0, 0));
            reference::start_selected(2, checkpoints.clone());
            reference::select_history(shared_id);
            reference::select_history(own_id);
            identities = Some((drone_id, generation, shared_id, own_id));
        }
        let (drone_id, generation, shared_id, own_id) = identities.unwrap();
        let freeze = crate::core::history_prediction::freeze_status();
        if freeze.0 != 0 {
            writeln!(
                &mut journal,
                "{}",
                serde_json::json!({"error": "frozen estimator", "freeze_status": freeze})
            )
            .unwrap();
            journal.flush().unwrap();
            journal.sync_data().unwrap();
            panic!("V7 estimator froze: {freeze:?}");
        }
        assert!(
            state.pop.voices.iter().any(|v| {
                v.id() == drone_id && v.metadata.generation == generation && v.is_alive()
            }),
            "long-lived drone disappeared"
        );
        churn_generation = churn_generation.max(
            state
                .pop
                .voices
                .iter()
                .filter(|v| v.id() != drone_id)
                .map(|v| v.metadata.generation)
                .max()
                .unwrap_or(0),
        );
        if let Some(id) = state
            .pop
            .voices
            .iter()
            .find(|v| v.id() != drone_id && v.is_alive())
            .map(|v| v.id())
            && last_churn_id != Some(id)
        {
            churn_births += u64::from(last_churn_id.is_some());
            last_churn_id = Some(id);
        }
        let own = state
            .schedule_renderer
            .self_sound_history(drone_id)
            .unwrap();
        // Exercise the drone's comparison learner without changing its phonation.
        // The render observes own PCM before shared PCM, so query only aligned clocks.
        if own.external.reference_identity().1 == observer.reference_identity().1
            && let Some(mut forecast) = observer.forecast()
        {
            observer.use_external_energy(&mut forecast, &mut own.external);
        }
        let (issues, shared_checkpoint, shared_counts) = reference::checkpoint_result(shared_id);
        let (_, own_checkpoint, own_counts) = reference::checkpoint_result(own_id);
        if heartbeat.elapsed() >= Duration::from_secs(60) {
            println!(
                "V7 progress shared_issues={issues} elapsed_sec={:.3}",
                start.elapsed().as_secs_f64()
            );
            heartbeat = Instant::now();
        }
        if shared_checkpoint != Some(checkpoints[next]) || own_checkpoint != shared_checkpoint {
            return;
        }
        let checkpoint = checkpoints[next];
        let counts = reference::counts();
        let elapsed = start.elapsed().as_secs_f64();
        let projected_full_sec = elapsed * 30_000_000.0 / checkpoint as f64;
        if full && next == 0 && projected_full_sec > 86400.0 {
            stop_after = 10_000_000;
        }
        let counts_json = |c: reference::Counts| {
            serde_json::json!({
                "passed": c.passed, "failed": c.failed, "inconclusive": c.inconclusive,
                "dd_inconclusive": c.dd_inconclusive, "kappa2_upper": c.condition_upper,
                "scaled_kappa2_upper": c.scaled_condition_upper,
                "uncertified_attempts": c.uncertified,
                "inconclusive_reason": "nonfinite certificate or interval overlaps tolerance; see reference log and saved values",
            })
        };
        let delta = reference::Counts {
            passed: std::array::from_fn(|i| counts.passed[i] - previous.passed[i]),
            failed: std::array::from_fn(|i| counts.failed[i] - previous.failed[i]),
            inconclusive: std::array::from_fn(|i| {
                counts.inconclusive[i] - previous.inconclusive[i]
            }),
            dd_inconclusive: counts.dd_inconclusive - previous.dd_inconclusive,
            uncertified: std::array::from_fn(|i| counts.uncertified[i] - previous.uncertified[i]),
            scaled_condition_upper: shared_counts
                .scaled_condition_upper
                .max(own_counts.scaled_condition_upper),
            condition_upper: shared_counts
                .condition_upper
                .max(own_counts.condition_upper),
        };
        let row = serde_json::json!({
            "checkpoint": checkpoint, "shared_issues_at_probe": issues,
            "elapsed_sec": elapsed, "score_sec": checkpoint as f64 / 100.0,
            "projected_full_sec": projected_full_sec, "stop_after": stop_after,
            "counts": counts_json(delta), "cumulative": counts_json(counts),
            "shared": {"id": shared_id, "counts": counts_json(shared_counts)},
            "own": {"id": own_id, "counts": counts_json(own_counts)},
            "drone_id": drone_id, "drone_generation": generation,
            "max_churn_generation": churn_generation,
            "churn_births": churn_births,
            "freeze_status": crate::core::history_prediction::freeze_status(),
        });
        serde_json::to_writer(&mut journal, &row).unwrap();
        writeln!(&mut journal).unwrap();
        journal.flush().unwrap();
        journal.sync_data().unwrap();
        println!("V7 checkpoint {row}");
        assert_eq!(
            std::array::from_fn::<_, 3, _>(|i| {
                delta.passed[i] + delta.failed[i] + delta.inconclusive[i]
            }),
            [1206, 1206, 603],
            "missing checkpoint outputs"
        );
        previous = counts;
        next += 1;
        if checkpoint == stop_after {
            *captured.lock().unwrap() = Some((counts, checkpoint, churn_births));
            reference::stop();
            stop_probe.store(true, Ordering::SeqCst);
        }
    });
    let mut config = AppConfig::default();
    config.analysis.hop_size = 256;
    let opts = WiringOptions {
        score: None,
        end_request: Arc::new(AtomicBool::new(false)),
        duration_sec: None,
        offline_body_probe: Some(probe),
        raw_t2_delay: Duration::ZERO,
        ui_tx: None,
        listener_forced: false,
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
    };
    let wiring =
        wire_runtime(&config, 48_000, "V7 precision".into(), scenario, stop, opts).unwrap();
    let worker = join_thread("V7 worker", wiring.worker_handle);
    let analysis = join_thread("V7 analysis", wiring.analysis_handle);
    worker.unwrap().unwrap();
    analysis.unwrap();
    if let Some(listener) = wiring.listener_analysis_handle {
        join_thread("V7 listener", listener).unwrap();
    }
    let (counts, checkpoint, churn) = result.lock().unwrap().expect("missing final checkpoint");
    counts.assert_passed();
    if checkpoint >= 2000 {
        assert!(churn > 0, "respawn population did not turn over");
    }
    println!(
        "V7 completed through issue={checkpoint}; full={}",
        checkpoint == 30_000_000
    );
}
