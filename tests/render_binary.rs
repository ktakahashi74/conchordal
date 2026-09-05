use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_path(ext: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "conchordal-render-it-{}-{nanos}.{ext}",
        std::process::id()
    ))
}

fn unique_wav_path() -> PathBuf {
    unique_temp_path("wav")
}

fn write_inline_scenario(script: &str) -> PathBuf {
    let path = unique_temp_path("rhai");
    fs::write(&path, script)
        .unwrap_or_else(|e| panic!("failed to write inline scenario {}: {e}", path.display()));
    path
}

fn run_render_binary(
    scenario: &std::path::Path,
    wav_path: &std::path::Path,
) -> std::process::Output {
    let exe = env!("CARGO_BIN_EXE_conchordal-render");
    let wav_path_str = wav_path.to_string_lossy().to_string();

    Command::new(exe)
        .env("RUST_LOG", "warn")
        .arg(scenario)
        .args(["-o", &wav_path_str])
        .output()
        .unwrap_or_else(|e| panic!("failed to run conchordal-render: {e}"))
}

#[test]
fn conchordal_render_generates_valid_wav() {
    let scenario = "tests/scripts/minimal_spawn_run.rhai";
    let wav_path = unique_wav_path();
    let output = run_render_binary(std::path::Path::new(scenario), &wav_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "conchordal-render failed: status={} stderr={stderr}",
            output.status
        );
    }

    let meta = fs::metadata(&wav_path)
        .unwrap_or_else(|e| panic!("expected output wav at {}: {e}", wav_path.display()));
    assert!(
        meta.len() > 44,
        "wav size too small (header only?): {} bytes",
        meta.len()
    );

    let reader =
        hound::WavReader::open(&wav_path).unwrap_or_else(|e| panic!("invalid wav output: {e}"));
    let spec = reader.spec();
    assert_eq!(spec.channels, 1, "render output must be mono");
    assert_eq!(
        spec.sample_format,
        hound::SampleFormat::Int,
        "render output must be PCM integer"
    );
    assert_eq!(spec.bits_per_sample, 16, "render output must be 16-bit");
    assert!(spec.sample_rate > 0, "invalid sample rate");
    assert!(reader.duration() > 0, "wav has no samples");
    let min_expected_samples = (spec.sample_rate as f32 * 0.8) as u32;
    assert!(
        reader.duration() >= min_expected_samples,
        "wav is too short for a 1s scenario: {} < {} samples",
        reader.duration(),
        min_expected_samples
    );

    let _ = fs::remove_file(&wav_path);
}

#[test]
fn conchordal_render_skips_zero_amp_note_on_and_still_writes_audio() {
    let scenario = write_inline_scenario(
        r#"
let silent = place(sine().anchor(), at(220.0)).amp(0.0);
let audible = place(sine().anchor(), at(330.0)).amp(0.25);
flush();
wait(0.4);
release(silent);
release(audible);
wait(0.2);
"#,
    );
    let wav_path = unique_wav_path();
    let output = run_render_binary(&scenario, &wav_path);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "expected render success for zero-amp NoteOn regression: status={} stderr={stderr}",
            output.status
        );
    }

    let meta = fs::metadata(&wav_path)
        .unwrap_or_else(|e| panic!("expected output wav at {}: {e}", wav_path.display()));
    assert!(
        meta.len() > 44,
        "wav size too small after zero-amp NoteOn regression scenario: {} bytes",
        meta.len()
    );

    let reader =
        hound::WavReader::open(&wav_path).unwrap_or_else(|e| panic!("invalid wav output: {e}"));
    assert!(reader.duration() > 0, "wav has no samples");
    // Duration alone passes on digital silence; require actual signal.
    let peak = reader
        .into_samples::<i16>()
        .filter_map(Result::ok)
        .map(|s| s.unsigned_abs())
        .max()
        .unwrap_or(0);
    assert!(peak > 0, "rendered wav is digital silence");

    let _ = fs::remove_file(&scenario);
    let _ = fs::remove_file(&wav_path);
}

#[test]
fn conchordal_render_report_matches_its_audio_and_preserves_output() {
    let scenario = write_inline_scenario(
        r#"
seed(73);
section("report fixture", || {
    let tone = place(sine().sustain().anchor().amp(0.1), at(220.0));
    wait(0.08);
    release(tone);
    wait(0.04);
});
"#,
    );
    let wav_path = unique_wav_path();
    let plain_wav_path = unique_wav_path();
    let report_path = unique_temp_path("jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .env("RUST_LOG", "warn")
        .arg(&scenario)
        .arg("-o")
        .arg(&wav_path)
        .arg("--report")
        .arg(&report_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "report render failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let records: Vec<serde_json::Value> = fs::read_to_string(&report_path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(records[0]["type"], "meta");
    assert_eq!(records[0]["seed"], 73);
    assert!(
        records[0]["hop_timing_scope"]
            .as_str()
            .unwrap()
            .contains("excludes hop_timing")
    );
    for record_type in ["scene_marker", "spawn", "listener_state", "rhythm_summary"] {
        assert!(
            records.iter().any(|record| record["type"] == record_type),
            "missing {record_type}"
        );
    }
    let timing: Vec<_> = records
        .iter()
        .filter(|record| record["type"] == "hop_timing")
        .collect();
    let reader = hound::WavReader::open(&wav_path).unwrap();
    let sample_rate = reader.spec().sample_rate as f64;
    let hop = 512;
    assert_eq!(reader.duration() as usize, timing.len() * hop);
    for (frame_idx, record) in timing.iter().enumerate() {
        assert_eq!(record["frame_idx"].as_u64(), Some(frame_idx as u64));
        let time_sec = record["time_sec"].as_f64().unwrap();
        assert!((time_sec * sample_rate - (frame_idx * hop) as f64).abs() < 1.0);
        let elapsed = record["elapsed_us"].as_f64().unwrap();
        let analysis_wait = record["analysis_wait_us"].as_f64().unwrap();
        let listener_wait = record["listener_wait_us"].as_f64().unwrap();
        assert!(elapsed >= analysis_wait + listener_wait);
        assert!(
            (record["hop_budget_us"].as_f64().unwrap() - hop as f64 / sample_rate * 1e6).abs()
                < 0.01
        );
        assert_eq!(record["audio_output"], "no_device");
        assert!(record["underrun_frames_total"].is_null());
    }
    let plain = run_render_binary(&scenario, &plain_wav_path);
    assert!(
        plain.status.success(),
        "plain render failed: {}",
        String::from_utf8_lossy(&plain.stderr)
    );
    assert_eq!(
        fs::read(&wav_path).unwrap(),
        fs::read(&plain_wav_path).unwrap(),
        "telemetry must not change the rendered waveform"
    );
    for path in [scenario, wav_path, plain_wav_path, report_path] {
        let _ = fs::remove_file(path);
    }
}

#[test]
fn conchordal_render_fails_when_report_cannot_be_created() {
    let wav_path = unique_wav_path();
    let report_path = unique_temp_path("dir");
    fs::create_dir(&report_path).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .env("RUST_LOG", "warn")
        .arg("tests/scripts/minimal_spawn_run.rhai")
        .arg("-o")
        .arg(&wav_path)
        .arg("--report")
        .arg(&report_path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("create report"));
    assert!(
        !wav_path.exists(),
        "invalid report destination must fail before WAV creation"
    );
    let _ = fs::remove_dir(report_path);
}

#[cfg(target_os = "linux")]
#[test]
fn conchordal_render_propagates_report_write_failures() {
    let wav_path = unique_wav_path();
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .env("RUST_LOG", "warn")
        .arg("tests/scripts/minimal_spawn_run.rhai")
        .arg("-o")
        .arg(&wav_path)
        .args(["--report", "/dev/full"])
        .output()
        .unwrap();
    assert!(
        !output.status.success(),
        "report I/O failures must fail the render"
    );
    assert!(String::from_utf8_lossy(&output.stderr).contains("report"));
    let _ = fs::remove_file(wav_path);
}

#[test]
fn conchordal_render_reserves_future_scenario_ids_before_respawning() {
    let scenario = write_inline_scenario(
        r#"
seed(1);
let turnover = place(
    sine().sustain().anchor().amp(0.02)
        .respawn_random().respawn_background_death_rate(1000000.0),
    at(220.0)
);
wait(0.1);
let later = place(sine().sustain().anchor().amp(0.02), at(330.0).count(32));
wait(0.1);
release(turnover);
release(later);
wait(0.1);
"#,
    );
    let wav_path = unique_wav_path();
    let output = run_render_binary(&scenario, &wav_path);
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "render failed: {stdout}\n{stderr}");
    assert!(
        !stdout.contains("id collision") && !stderr.contains("id collision"),
        "runtime respawns must not consume a later Population's Voice IDs: {stdout}\n{stderr}"
    );
    let _ = fs::remove_file(&scenario);
    let _ = fs::remove_file(&wav_path);
}

#[test]
fn conchordal_render_matches_preintervention_audio_with_reserved_runtime_ids() {
    let source = r#"
seed(73);
let turnover = place(
    sine().brain("drone").sustain().anchor().amp(0.1)
        .respawn_random().respawn_background_death_rate(40.0),
    at(220.0)
);
wait(0.4);
let later = place(sine().sustain().anchor().amp(0.02), at(330.0).count(9));
wait(0.08);
release(turnover);
release(later);
wait(0.08);
"#;
    let with_later = write_inline_scenario(source);
    let without_later = write_inline_scenario(
        &source
            .replace(
                "let later = place(sine().sustain().anchor().amp(0.02), at(330.0).count(9));\n",
                "",
            )
            .replace("release(later);\n", ""),
    );
    let render = |scenario: &std::path::Path, reservation: Option<u64>| {
        let wav = unique_wav_path();
        let report = unique_temp_path("jsonl");
        let mut command = Command::new(env!("CARGO_BIN_EXE_conchordal-render"));
        command
            .env("RUST_LOG", "warn")
            .arg(scenario)
            .arg("-o")
            .arg(&wav)
            .arg("--report")
            .arg(&report);
        if let Some(bound) = reservation {
            command
                .arg("--reserve-runtime-ids-through")
                .arg(bound.to_string());
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "render failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let reader = hound::WavReader::open(&wav).unwrap();
        let prefix_frames = (reader.spec().sample_rate as f64 * 0.4) as usize;
        let samples: Vec<i16> = reader.into_samples().map(Result::unwrap).collect();
        assert!(samples.len() >= prefix_frames);
        let respawns: Vec<serde_json::Value> = fs::read_to_string(&report)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|record| {
                record["type"] == "respawn" && record["time_sec"].as_f64().unwrap() < 0.4
            })
            .collect();
        assert!(
            !respawns.is_empty(),
            "fixture must respawn before intervention"
        );
        for path in [wav, report] {
            let _ = fs::remove_file(path);
        }
        (samples, prefix_frames, respawns)
    };
    let control = render(&with_later, None);
    let smaller_bound = render(&with_later, Some(1));
    let matched = render(&without_later, Some(10));
    let unreserved = render(&without_later, None);
    assert_eq!(
        control.0, smaller_bound.0,
        "a smaller bound must not shrink scenario reservations"
    );
    assert_eq!(control.2, smaller_bound.2);
    assert_eq!(control.1, matched.1);
    let prefix = &control.0[..control.1];
    assert!(prefix.iter().any(|&sample| sample != 0));
    assert_eq!(prefix, &matched.0[..matched.1]);
    assert_eq!(
        control.2, matched.2,
        "matched IDs must preserve respawn events"
    );
    assert_eq!(control.2[0]["voice_id"], 11);
    assert_eq!(unreserved.2[0]["voice_id"], 2);
    assert_ne!(prefix, &unreserved.0[..unreserved.1]);
    for path in [with_later, without_later] {
        let _ = fs::remove_file(path);
    }
}

#[test]
fn conchordal_render_rejects_exhausted_runtime_id_reservation() {
    let wav = unique_wav_path();
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .arg("tests/scripts/minimal_spawn_run.rhai")
        .arg("-o")
        .arg(&wav)
        .arg("--reserve-runtime-ids-through")
        .arg(u64::MAX.to_string())
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(!wav.exists());
    let error = conchordal::app::run_render(
        "tests/scripts/minimal_spawn_run.rhai",
        wav.to_string_lossy().into_owned(),
        conchordal::config::AppConfig::default(),
        std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
        None,
        None,
        u64::MAX,
    )
    .unwrap_err();
    assert!(error.contains("reserve-runtime-ids-through"));
    assert!(!wav.exists());
}

#[test]
fn conchordal_render_fails_when_wav_thread_panics() {
    let scenario = write_inline_scenario(
        r#"
let tone = place(sine().anchor(), at(220.0)).amp(0.2);
flush();
wait(0.2);
release(tone);
wait(0.1);
"#,
    );
    let bad_output = unique_temp_path("dir");
    fs::create_dir_all(&bad_output).unwrap_or_else(|e| {
        panic!(
            "failed to create output directory {}: {e}",
            bad_output.display()
        )
    });

    let output = run_render_binary(&scenario, &bad_output);
    assert!(
        !output.status.success(),
        "expected render failure when wav thread panics; stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );

    let _ = fs::remove_file(&scenario);
    let _ = fs::remove_dir_all(&bad_output);
}
