//! Native deterministic thread delivery controls for the opt-in meter.
use super::*;
use std::io::{BufRead, BufReader};

use super::test_output_directory::TestOutputDirectory;

#[test]
fn continuous_offline_delivery_is_independent_of_producer_delay() {
    let directory = std::env::temp_dir().join(format!(
        "conchordal-l2-thread-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    let guard = TestOutputDirectory::new(directory.clone());
    let script = directory.join("input.rhai");
    std::fs::write(
        &script,
        include_str!("../../tests/fixtures/temporal_meter_t2.rhai"),
    )
    .unwrap();
    let config = AppConfig {
        temporal_meter_t2: true,
        ..AppConfig::default()
    };
    config.validate().unwrap();
    let mut reference = None;
    for (name, delay) in [
        ("normal", Duration::ZERO),
        ("delayed", Duration::from_millis(2)),
    ] {
        let path = directory.join(format!("{name}.jsonl"));
        let scenario = compile_scenario_from_script(&script, None).unwrap();
        let mut reporter = JsonlReporter::create(path.to_str().unwrap()).unwrap();
        reporter.write_meta(scenario.seed).unwrap();
        let wiring = wire_runtime(
            &config,
            config.audio.sample_rate,
            "L2 thread control".into(),
            scenario,
            Arc::new(AtomicBool::new(false)),
            WiringOptions {
                offline_body_probe: None,
                raw_t2_delay: delay,
                ui_tx: None,
                listener_forced: true,
                wait_user_exit: false,
                start_playing: true,
                audio_prod: None,
                wav_tx: None,
                reporter: Some(reporter),
                deterministic_analysis: true,
                deterministic_footprints: true,
                guard_meter: None,
                underrun_frames: None,
                reserve_runtime_ids_through: 0,
                profile: None,
                audio_counters: None,
            },
        )
        .unwrap();
        join_thread("worker", wiring.worker_handle)
            .unwrap()
            .unwrap();
        join_thread("analysis", wiring.analysis_handle).unwrap();
        if let Some(handle) = wiring.listener_analysis_handle {
            join_thread("listener", handle).unwrap();
        }
        let mut rows = Vec::new();
        for line in BufReader::new(std::fs::File::open(path).unwrap()).lines() {
            let mut row: serde_json::Value = serde_json::from_str(&line.unwrap()).unwrap();
            if row["type"] == "hop_timing" {
                continue;
            }
            if row["type"] == "temporal_meter" {
                row["input"]
                    .as_object_mut()
                    .unwrap()
                    .remove("processing_us");
            }
            rows.push(row);
        }
        assert!(rows.iter().any(|r| r["type"] == "temporal_meter"));
        if let Some(ref expected) = reference {
            assert_eq!(&rows, expected);
        } else {
            reference = Some(rows);
        }
    }
    drop(guard);
}

#[test]
fn continuous_runtime_rejects_actual_device_rate_before_spawning() {
    let config = AppConfig {
        temporal_meter_t2: true,
        ..AppConfig::default()
    };
    let result = wire_runtime(
        &config,
        8000,
        "invalid raw rate".into(),
        Scenario {
            expression: None,
            expression_cursor: None,
            seed: 21,
            output_gain_db: 0.0,
            output_gain_position: None,
            temporal_mode: Default::default(),
            control_update_mode: Default::default(),
            scaffold: Default::default(),
            meter_shaping: Default::default(),
            scene_markers: Vec::new(),
            events: Vec::new(),
            duration_sec: 0.0,
        },
        Arc::new(AtomicBool::new(false)),
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
            deterministic_analysis: false,
            deterministic_footprints: false,
            guard_meter: None,
            underrun_frames: None,
            reserve_runtime_ids_through: 0,
            profile: None,
            audio_counters: None,
        },
    );
    assert!(matches!(result,Err(reason) if reason.contains("Nyquist")));
}
