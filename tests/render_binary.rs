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
fn temporal_observation_preserves_audio_and_finishes_both_buses() {
    let config = unique_temp_path("toml");
    fs::write(
        &config,
        "[audio]\nsample_rate = 48000\n[dcc]\ncoupling_strength = 0.0\n",
    )
    .unwrap();
    let body = r#"
seed(9122026);
section("author label is not an observation", || {
    let tone = place(sine().sustain().anchor().amp(0.1).send(presentation_bus), at(220.0));
    wait(0.4);
    release(tone);
    wait(0.2);
});
"#;
    let mut reference_audio = None;
    for (mode, ridge_enabled, acoustic_enabled, period_model) in [
        ("off", false, false, None),
        ("observe", false, false, None),
        ("observe", true, false, None),
        ("observe", true, true, None),
        (
            "observe",
            true,
            true,
            Some(conchordal::config::ArrivalModel::Hazard),
        ),
        (
            "observe",
            true,
            true,
            Some(conchordal::config::ArrivalModel::Periodic),
        ),
    ] {
        fs::write(
            &config,
            if ridge_enabled {
                "[audio]\nsample_rate = 48000\n[dcc]\ncoupling_strength = 0.0\n[temporal_ridge]\nmeans = [0.0, 0.0, 0.0]\ndeviations = [0.05, 4.0, 1.0]\n"
            } else {
                "[audio]\nsample_rate = 48000\n[dcc]\ncoupling_strength = 0.0\n"
            },
        ).unwrap();
        if acoustic_enabled {
            use std::io::Write;
            let mut file = fs::OpenOptions::new().append(true).open(&config).unwrap();
            file.write_all(b"[temporal_acoustic]\ngroup_means = [0.0, 0.0, 0.0]\ngroup_deviations = [0.05, 4.0, 1.0]\naccent_means = [0.0, 0.0]\naccent_deviations = [1.0, 1.0]\ngroup_retirement_sec = 2.0\ninactive_energy_max = 1e-8\ncorrelation_window_sec = 0.25\nmin_pairs = 8\nmin_coverage = 0.9\npersistence_hops = 3\n").unwrap();
        }

        if let Some(model) = period_model {
            use std::io::Write;
            let period = conchordal::config::TemporalPeriodConfig {
                model,
                coefficients: [0.; 18],
                means: [0.; 8],
                deviations: [1.; 8],
                horizon_sec: 0.1,
            };
            let text = format!("[temporal_period]\n{}", toml::to_string(&period).unwrap());
            fs::OpenOptions::new()
                .append(true)
                .open(&config)
                .unwrap()
                .write_all(text.as_bytes())
                .unwrap();
        }

        for reporting in [false, true] {
            let scenario = write_inline_scenario(&format!("temporal_mode(\"{mode}\");\n{body}"));
            let wav = unique_wav_path();
            let report = unique_temp_path("jsonl");
            let mut command = Command::new(env!("CARGO_BIN_EXE_conchordal-render"));
            command
                .env("RUST_LOG", "warn")
                .arg(&scenario)
                .arg("-o")
                .arg(&wav)
                .arg("--config")
                .arg(&config);
            if reporting {
                command.arg("--report").arg(&report);
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{mode}, report={reporting}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
            let audio = fs::read(&wav).unwrap();
            if let Some(reference) = &reference_audio {
                assert_eq!(
                    &audio, reference,
                    "{mode}, report={reporting} changed audio"
                );
            } else {
                reference_audio = Some(audio);
            }
            if reporting {
                let records: Vec<serde_json::Value> = fs::read_to_string(&report)
                    .unwrap()
                    .lines()
                    .map(|s| serde_json::from_str(s).unwrap())
                    .collect();
                let observations: Vec<_> = records
                    .iter()
                    .filter(|r| r["type"] == "temporal_observation")
                    .collect();
                if mode == "off" {
                    assert!(observations.is_empty());
                } else {
                    let hops = records.iter().filter(|r| r["type"] == "hop_timing").count() as u64;
                    assert!(hops > 32);
                    for bus in 0..2 {
                        let last = &observations
                            .iter()
                            .rev()
                            .find(|r| r["observation"]["bus"] == bus)
                            .unwrap()["observation"];
                        assert_eq!(last["state"], "finished");
                        assert_eq!(last["received_frames"], hops);
                        assert_eq!(last["frame_id"], hops - 1);
                        assert_eq!(last["support_end_sample"], hops * 512);
                        assert_eq!(last["available_sample"], hops * 512);
                        assert_eq!(last["hop_start_sample"], (hops - 1) * 512);
                        assert_eq!(last["sample_rate"], 48000);
                        assert_eq!(last["source_missing_samples"], 0);
                        assert_eq!(last["delivery_dropped_frames"], 0);
                        assert_eq!(last["rejected_frames"], 0);
                        assert_eq!(last["relations_implemented"], false);
                        assert_eq!(last["action_enabled"], false);
                        assert_eq!(last["ridge_failed"], false);
                        assert!(last["period_error"].is_null(), "{last}");
                        assert_eq!(last["period"].is_object(), period_model.is_some());
                        assert_eq!(
                            last["period_parameters"].is_object(),
                            period_model.is_some()
                        );
                        assert_eq!(last["ridge_parameters"].is_object(), ridge_enabled);
                        assert_eq!(last["ridges"].is_object(), ridge_enabled);
                        assert!(last["fully_supported_frames"].as_u64().unwrap() > 0);
                        let mono = last["mono_mean_square"].as_f64().unwrap();
                        let energies = last["trajectories"]["energy"].as_array().unwrap();
                        let total: f64 = energies.iter().map(|v| v.as_f64().unwrap()).sum();
                        assert!((total - mono).abs() < 1e-13);
                        if bus == 0 {
                            assert_eq!(mono, 0.0);
                            assert!(energies.iter().all(|v| v == 0.0));
                        }
                    }
                    assert!(
                        observations
                            .iter()
                            .all(|r| !r.to_string().contains("author label"))
                    );
                    let known_habitat: Vec<_> = observations
                        .iter()
                        .filter(|r| r["observation"]["bus"] == 0)
                        .filter_map(|r| r["observation"]["spectral_power_sum"].as_f64())
                        .collect();
                    assert!(!known_habitat.is_empty());
                    assert!(known_habitat.iter().all(|&power| power == 0.0));
                    assert!(
                        observations
                            .iter()
                            .filter(|r| r["observation"]["bus"] == 1)
                            .any(|r| r["observation"]["spectral_power_sum"]
                                .as_f64()
                                .is_some_and(|power| power > 0.0))
                    );

                    if acoustic_enabled {
                        let mut group_features = 0;
                        for record in &observations {
                            let o = &record["observation"];
                            assert_eq!(o["acoustic_failed"], false);
                            let a = &o["acoustic"];
                            if let Some(energy) = a["energy"].as_array() {
                                let sum: f64 = energy.iter().map(|v| v.as_f64().unwrap()).sum();
                                assert!(
                                    (sum - o["mono_mean_square"].as_f64().unwrap()).abs() < 1e-12
                                );
                            }
                            if let Some(features) = a["features"].as_array() {
                                for (slot, feature) in
                                    features.iter().enumerate().filter(|(_, f)| f.is_object())
                                {
                                    let raw = &feature["raw"];
                                    assert_eq!(raw["group"], a["group_handles"][slot]);
                                    assert_eq!(raw["group"]["bus"], o["bus"]);
                                    assert_eq!(raw["source_end"], o["support_end_sample"]);
                                    if slot < 7 {
                                        assert_eq!(
                                            o["bus"], 1,
                                            "silent habitat admitted a sound group"
                                        );
                                        group_features += 1;
                                    }
                                }
                            }
                        }
                        assert!(
                            group_features > 0,
                            "render did not reach sound group descriptors"
                        );
                    }
                    if ridge_enabled {
                        let mut links = 0;
                        for record in &observations {
                            let observation = &record["observation"];
                            if let Some(current) = observation["ridges"]["current"].as_array() {
                                for ridge in current.iter().filter(|r| r.is_object()) {
                                    assert_eq!(ridge["handle"]["bus"], observation["bus"]);
                                    assert_eq!(
                                        ridge["handle"]["epoch"],
                                        observation["source_epoch"]
                                    );
                                    assert_eq!(
                                        ridge["end_sample"],
                                        observation["support_end_sample"]
                                    );
                                    assert_eq!(
                                        observation["bus"], 1,
                                        "silent bus produced a ridge"
                                    );
                                    for link in ridge["links"]
                                        .as_array()
                                        .unwrap()
                                        .iter()
                                        .filter(|l| l.is_object())
                                    {
                                        assert!(
                                            link["parent_end_sample"].as_u64().unwrap()
                                                < ridge["end_sample"].as_u64().unwrap()
                                        );
                                        links += 1;
                                    }
                                }
                            }
                        }
                        assert!(
                            links > 0,
                            "actual rendered audio never reached ridge continuity"
                        );
                    }
                }
                fs::remove_file(report).unwrap();
            }
            fs::remove_file(scenario).unwrap();
            fs::remove_file(wav).unwrap();
        }
    }
    fs::remove_file(config).unwrap();
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

#[test]
fn body_prototypes_match_actual_descriptors_without_changing_audio() {
    use conchordal::config::{
        AppConfig, ArrivalModel, TemporalAcousticConfig, TemporalBodyConfig, TemporalBodyMedoid,
        TemporalBodyPrototypesConfig, TemporalPeriodConfig, TemporalRidgeConfig,
    };
    let mut config = AppConfig::default();
    config.analysis.nfft = 2048;
    config.analysis.hop_size = 512;
    config.audio.sample_rate = 48000;
    config.dcc.coupling_strength = 0.;
    let scales = TemporalBodyConfig {
        means: [0.; 6],
        deviations: [1.; 6],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
    };
    config.temporal_body = Some(scales);
    config.temporal_ridge = Some(TemporalRidgeConfig {
        means: [0.; 3],
        deviations: [0.05, 4., 1.],
    });
    config.temporal_acoustic = Some(TemporalAcousticConfig {
        group_means: [0.; 3],
        group_deviations: [0.05, 4., 1.],
        accent_means: [0.; 2],
        accent_deviations: [1.; 2],
        group_retirement_sec: 2.,
        inactive_energy_max: 1e-8,
        correlation_window_sec: 0.25,
        min_pairs: 8,
        min_coverage: 0.9,
        persistence_hops: 3,
    });
    config.temporal_period = Some(TemporalPeriodConfig {
        model: ArrivalModel::Hazard,
        coefficients: [0.; 18],
        means: [0.; 8],
        deviations: [1.; 8],
        horizon_sec: 0.1,
    });
    let run = |config: &AppConfig, mode: &str, reporting: bool| {
        let config_path = unique_temp_path("toml");
        fs::write(&config_path, toml::to_string(config).unwrap()).unwrap();
        let script = write_inline_scenario(&format!(
            r#"
temporal_mode("{mode}");
seed(20260917);
let population = place(sine().sustain().anchor().amp(0.1).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(220.0));
wait(1.0);
release(population);
wait(0.3);
"#
        ));
        let wav = unique_wav_path();
        let report = unique_temp_path("jsonl");
        let mut command = Command::new(env!("CARGO_BIN_EXE_conchordal-render"));
        command
            .env("RUST_LOG", "warn")
            .arg(&script)
            .arg("--config")
            .arg(&config_path)
            .arg("-o")
            .arg(&wav);
        if reporting {
            command.arg("--report").arg(&report);
        }
        let result = command.output().unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        let audio = fs::read(&wav).unwrap();
        let records: Vec<serde_json::Value> = if reporting {
            fs::read_to_string(&report)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str(line).unwrap())
                .collect()
        } else {
            Vec::new()
        };
        for path in [&config_path, &script, &wav] {
            fs::remove_file(path).unwrap();
        }
        if reporting {
            fs::remove_file(report).unwrap();
        }
        (audio, records)
    };
    let (baseline, records) = run(&config, "observe", true);
    let source = records
        .iter()
        .find(|r| {
            r["type"] == "body_descriptor"
                && r["bus"] == 1
                && r["mask"] == 63
                && r["end"].as_u64().unwrap() >= 9600
        })
        .unwrap();
    let raw_values = std::array::from_fn(|i| source["raw_values"][i].as_f64().unwrap());
    let end = source["end"].as_u64().unwrap();
    assert!(source["prototype_assignment"].is_null());
    config.temporal_body_prototypes = Some(TemporalBodyPrototypesConfig {
        model_version: "34".repeat(32),
        sample_rate: 48000,
        nfft: 2048,
        hop_size: 512,
        means: scales.means,
        deviations: scales.deviations,
        accent_means: scales.accent_means,
        accent_deviations: scales.accent_deviations,
        medoids: vec![TemporalBodyMedoid {
            record_id: "actual-rendered-record".into(),
            raw_values,
            mask: 63,
        }],
    });
    for mode in ["off", "observe"] {
        for reporting in [false, true] {
            let (audio, records) = run(&config, mode, reporting);
            assert_eq!(audio, baseline, "{mode}, reporting={reporting}");
            if !reporting {
                continue;
            }
            let descriptors: Vec<_> = records
                .iter()
                .filter(|r| r["type"] == "body_descriptor")
                .collect();
            if mode == "off" {
                assert!(descriptors.is_empty());
                continue;
            }
            let matched = descriptors
                .iter()
                .find(|r| r["bus"] == 1 && r["end"] == end)
                .unwrap();
            assert_eq!(matched["raw_values"], source["raw_values"]);
            assert_eq!(matched["source_id"], source["source_id"]);
            assert_eq!(matched["body_generation"], source["body_generation"]);
            assert_eq!(
                matched["prototype_assignment"]["key"],
                serde_json::json!([0, 0])
            );
            // The report-JSON -> TOML model round trip can move a descriptor by one ulp.
            assert!(
                matched["prototype_assignment"]["distance"]
                    .as_f64()
                    .unwrap()
                    < 1e-14
            );
            assert_eq!(matched["prototype_assignment"]["common_coordinates"], 6);
            assert!(
                descriptors
                    .iter()
                    .filter(|r| r["bus"] == 0)
                    .all(|r| r["prototype_assignment"].is_null())
            );
            let snapshot = records
                .iter()
                .rev()
                .find(|r| r["type"] == "body_observation")
                .unwrap();
            assert_eq!(
                snapshot["prototype_model_version"],
                serde_json::json!([0x34; 32].to_vec())
            );
            assert_eq!(snapshot["finished"], true);
            assert_eq!(snapshot["capture_drops"], 0);
            let mut group_matches = 0;
            let mut distinct_cuts = std::collections::BTreeSet::new();
            for observation in records
                .iter()
                .filter(|r| r["type"] == "temporal_observation")
                .map(|r| &r["observation"])
            {
                let context = &observation["group_prototypes"];
                if context.is_null() {
                    continue;
                }
                assert_eq!(context["bus"], observation["bus"]);
                assert_eq!(context["epoch"], observation["source_epoch"]);
                assert!(
                    context["end_sample"].as_u64().unwrap()
                        <= observation["support_end_sample"].as_u64().unwrap()
                );
                assert_eq!(
                    context["model_version"],
                    serde_json::json!([0x34; 32].to_vec())
                );
                if context["bus"] == 1 {
                    distinct_cuts.insert(context["end_sample"].as_u64().unwrap());
                }
                for assignment in context["assignments"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|a| !a.is_null())
                {
                    assert_eq!(
                        context["bus"], 1,
                        "presentation prototype must not match the silent habitat"
                    );
                    let descriptor = context["descriptors"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .find(|d| {
                            d["group"]["epoch"] == assignment["key"][0]
                                && d["group"]["generation"] == assignment["key"][1]
                        })
                        .unwrap();
                    assert_eq!(descriptor["group"]["bus"], context["bus"]);
                    assert_eq!(descriptor["end"], context["end_sample"]);
                    assert!(
                        descriptor["available"].as_u64().unwrap()
                            <= descriptor["end"].as_u64().unwrap()
                    );
                    assert!(
                        descriptor["end"].as_u64().unwrap() - descriptor["start"].as_u64().unwrap()
                            <= 96000
                    );
                    assert!(assignment["distance"].as_f64().unwrap() <= 0.25);
                    group_matches += 1;
                }
            }
            assert!(
                group_matches > 0,
                "actual shared audio never matched its private-audio prototype"
            );
            assert!(
                distinct_cuts
                    .iter()
                    .zip(distinct_cuts.iter().skip(1))
                    .all(|(a, b)| b - a >= 4800)
            );
        }
    }
}

#[test]
fn private_body_descriptors_use_real_routes_and_do_not_change_audio() {
    let config = unique_temp_path("toml");
    fs::write(
        &config,
        r#"
[audio]
sample_rate = 48000
[analysis]
nfft = 2048
hop_size = 512
[dcc]
coupling_strength = 0.0
[temporal_body]
means = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
deviations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
accent_means = [0.0, 0.0]
accent_deviations = [1.0, 1.0]
"#,
    )
    .unwrap();
    let mut reference = None;
    for mode in ["off", "observe"] {
        for reporting in [false, true] {
            let scenario = write_inline_scenario(&format!(
                r#"
temporal_mode("{mode}");
seed(20260917);
let tone = place(sine().sustain().anchor().amp(0.1).adsr(0.01, 0.01, 1.0, 0.3).send(presentation_bus), at(220.0));
wait(1.0);
release(tone);
wait(2.1);
"#
            ));
            let wav = unique_wav_path();
            let report = unique_temp_path("jsonl");
            let mut command = Command::new(env!("CARGO_BIN_EXE_conchordal-render"));
            command
                .env("RUST_LOG", "warn")
                .arg(&scenario)
                .arg("-o")
                .arg(&wav)
                .arg("--config")
                .arg(&config);
            if reporting {
                command.arg("--report").arg(&report);
            }
            let output = command.output().unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let audio = fs::read(&wav).unwrap();
            if let Some(reference) = &reference {
                assert_eq!(&audio, reference);
            } else {
                reference = Some(audio);
            }
            if reporting {
                let records: Vec<serde_json::Value> = fs::read_to_string(&report)
                    .unwrap()
                    .lines()
                    .map(|l| serde_json::from_str(l).unwrap())
                    .collect();
                let descriptors: Vec<_> = records
                    .iter()
                    .filter(|r| r["type"] == "body_descriptor")
                    .collect();
                if mode == "off" {
                    assert!(descriptors.is_empty());
                    assert!(!records.iter().any(|r| r["type"] == "body_observation"));
                } else {
                    assert!(!descriptors.is_empty());
                    let final_state = records
                        .iter()
                        .rev()
                        .find(|r| r["type"] == "body_observation")
                        .unwrap();
                    assert_eq!(final_state["finished"], true);
                    assert_eq!(final_state["capture_drops"], 0);
                    assert_eq!(final_state["invalid_hops"], 0);
                    assert_eq!(final_state["outside_voice_hops"], 0);
                    assert!(
                        descriptors
                            .iter()
                            .any(|r| r["bus"] == 1 && r["mask"].as_u64().unwrap() & 3 == 3)
                    );
                    for record in descriptors {
                        assert!(
                            record["end"].as_u64().unwrap() > record["start"].as_u64().unwrap()
                        );
                        assert!(
                            record["available"].as_u64().unwrap()
                                >= record["end"].as_u64().unwrap()
                        );
                        if record["bus"] == 0 {
                            assert_eq!(record["mask"].as_u64().unwrap() & 3, 0);
                            assert!(
                                (record["raw_values"][2].as_f64().unwrap() - 1e-6_f64.log2()).abs()
                                    < 1e-12
                            );
                        }
                    }
                }
            }
            fs::remove_file(scenario).unwrap();
            fs::remove_file(wav).unwrap();
            if reporting {
                fs::remove_file(report).unwrap();
            }
        }
    }
    fs::remove_file(config).unwrap();
}

#[test]
fn footprint_only_renders_repeat_exactly_without_observe_or_body_analysis() {
    let config = unique_temp_path("toml");
    fs::write(
        &config,
        r#"
[audio]
sample_rate = 48000
[analysis]
nfft = 2048
hop_size = 512
[dcc]
coupling_strength = 0.0
[temporal_onset_comparison]
footprint = "body"
"#,
    )
    .unwrap();
    let scenario = write_inline_scenario(
        r#"
seed(20260923);
let voice = modal().amp(0.03).flow().cycles(3).adsr(0.03, 0.3, 0.5, 0.7)
    .send(habitat_bus | presentation_bus);
let group = place(voice, consonance(220.0, 880.0).spacing(0.8).count(4));
wait(4.0);
release(group);
wait(1.0);
"#,
    );
    let mut runs = Vec::new();
    for _ in 0..2 {
        let wav = unique_wav_path();
        let report = unique_temp_path("jsonl");
        let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
            .env("RUST_LOG", "warn")
            .arg(&scenario)
            .arg("-o")
            .arg(&wav)
            .arg("--config")
            .arg(&config)
            .arg("--report")
            .arg(&report)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let all_records: Vec<serde_json::Value> = fs::read_to_string(&report)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        for kind in [
            "temporal_observation",
            "body_observation",
            "body_descriptor",
            "body_candidate_energy",
            "body_default",
            "self_sound_observation",
            "self_sound_outcome",
            "self_sound_descriptor_prediction",
        ] {
            assert!(
                all_records.iter().all(|record| record["type"] != kind),
                "unexpected analysis: {kind}"
            );
        }
        let stats = all_records
            .iter()
            .find(|record| record["type"] == "body_footprint_worker")
            .unwrap();
        assert!(stats["footprint_requested"].as_u64().unwrap() > 0);
        assert_eq!(stats["footprint_requested"], stats["footprint_completed"]);
        assert_eq!(stats["worker_failed"], false);
        // Everything a decision can depend on; `computed_at` is a wall-clock field.
        let records: Vec<serde_json::Value> = fs::read_to_string(&report)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
            .filter(|record| {
                matches!(
                    record["type"].as_str(),
                    Some(
                        "participation_decision"
                            | "participation_context"
                            | "body_footprint"
                            | "onset"
                    )
                )
            })
            .map(|mut record| {
                record.as_object_mut().unwrap().remove("computed_at");
                record
            })
            .collect();
        runs.push((fs::read(&wav).unwrap(), records));
    }
    let (first, second) = (&runs[0], &runs[1]);
    assert!(first.1.iter().any(|r| r["type"] == "body_footprint"));
    assert!(first.1.iter().any(|r| r["footprint_source"] == "body"));
    assert!(
        first.0 == second.0,
        "a body render changed its audio on repeat"
    );
    assert_eq!(first.1, second.1);

    let proxy_config = fs::read_to_string(&config)
        .unwrap()
        .replace("footprint = \"body\"", "footprint = \"proxy\"");
    fs::write(&config, proxy_config).unwrap();
    let proxy_wav = unique_wav_path();
    let proxy_report = unique_temp_path("jsonl");
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .env("RUST_LOG", "warn")
        .arg(&scenario)
        .arg("-o")
        .arg(&proxy_wav)
        .arg("--config")
        .arg(&config)
        .arg("--report")
        .arg(&proxy_report)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_ne!(
        first.0,
        fs::read(&proxy_wav).unwrap(),
        "body and proxy audio match"
    );
    let proxy_records: Vec<serde_json::Value> = fs::read_to_string(&proxy_report)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let mut body_decisions: Vec<_> = first
        .1
        .iter()
        .filter(|r| r["type"] == "participation_decision" && r["footprint_source"] == "body")
        .collect();
    body_decisions.sort_by_key(|r| (r["now"].as_u64().unwrap(), r["voice_id"].as_u64().unwrap()));
    let mut compared = 0;
    let mut found_divergence = false;
    for body in body_decisions {
        let Some(proxy) = proxy_records.iter().find(|r| {
            r["type"] == "participation_decision"
                && r["footprint_source"] == "proxy(setting)"
                && r["voice_id"] == body["voice_id"]
                && r["now"] == body["now"]
        }) else {
            panic!("proxy decision missing before the first selection divergence");
        };
        for field in [
            "due_frame",
            "period_frames",
            "memory",
            "own_band_energy",
            "forecast_observed_frame",
            "current_identity",
        ] {
            assert_eq!(
                proxy[field], body[field],
                "{field} differs before selection"
            );
        }
        assert_eq!(proxy["footprint_identity"], body["footprint_identity"]);
        assert_eq!(
            proxy["footprint_received_at"],
            body["footprint_received_at"]
        );
        assert!(proxy_records.iter().any(|r| {
            r["type"] == "body_footprint"
                && r["identity"] == proxy["footprint_identity"]
                && r["received_at"] == proxy["footprint_received_at"]
        }));
        assert_eq!(proxy["footprint_d_samples"], body["footprint_d_samples"]);
        assert_eq!(proxy["footprint_delay"], body["footprint_delay"]);
        for (body_candidate, proxy_candidate) in body["candidates"]
            .as_array()
            .unwrap()
            .iter()
            .zip(proxy["candidates"].as_array().unwrap())
        {
            assert_eq!(proxy_candidate["at"], body_candidate["at"]);
            assert_eq!(
                proxy_candidate["external_energy"],
                body_candidate["external_energy"]
            );
        }
        compared += 1;
        if proxy["selected_offset"] != body["selected_offset"]
            || proxy["skipped"] != body["skipped"]
        {
            assert_ne!(proxy["footprint_power"], body["footprint_power"]);
            found_divergence = true;
            break;
        }
    }
    assert!(
        compared > 0,
        "no matching body/proxy decisions used a delivered record"
    );
    assert!(
        found_divergence,
        "modal body/proxy selections never diverged"
    );
    fs::remove_file(config).unwrap();
    fs::remove_file(scenario).unwrap();
    fs::remove_file(proxy_wav).unwrap();
    fs::remove_file(proxy_report).unwrap();
}
