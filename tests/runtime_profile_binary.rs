use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn fixture(dcc: bool) -> PathBuf {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("conchordal-profile-{}-{stamp}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    fs::write(
        dir.join("scenario.rhai"),
        r#"
seed(42);
let voice = place(sine().brain("drone").anchor().sustain().amp(0.01)
    .adsr(0.01, 0.01, 1.0, 0.02), at(220.0));
wait(0.05);
release(voice);
wait(0.05);
"#,
    )
    .unwrap();
    fs::write(
        dir.join("config.toml"),
        format!(
            "[analysis]\nnfft = 1024\nhop_size = 512\n[dcc]\ncoupling_strength = {}\n",
            if dcc { "0.5" } else { "0.0" },
        ),
    )
    .unwrap();
    dir
}

fn profile_command(dir: &Path, path: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_conchordal"));
    cmd.env("RUST_LOG", "warn")
        .env_remove("CONCHORDAL_LIMITER")
        .arg(dir.join("scenario.rhai"))
        .arg("--config")
        .arg(dir.join("config.toml"))
        .args(["--nogui", "--play=false", "--profile"])
        .arg(path);
    cmd
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn device_free_profile_is_independent_of_report_and_tracks_effective_listener() {
    for dcc in [false, true] {
        for report in [false, true] {
            let dir = fixture(dcc);
            let path = dir.join("profile.json");
            let mut cmd = profile_command(&dir, &path);
            if report {
                cmd.arg("--report").arg(dir.join("report.jsonl"));
            }
            assert_success(&cmd.output().unwrap());
            let profile: serde_json::Value =
                serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
            assert_eq!(profile["schema_version"], 2);
            assert_eq!(profile["seed"], 42);
            assert_eq!(profile["report_enabled"], report);
            assert_eq!(profile["listener_enabled"], dcc || report);
            assert_eq!(profile["audio_output"], "no_device");
            assert!(profile["audio"].is_null());
            assert_eq!(profile["truncated"], false);
            assert_eq!(profile["dropped_hops"], 0);
            assert_eq!(
                profile["allocation_instrumented"],
                cfg!(feature = "profile-alloc")
            );
            let hops = profile["hops"].as_array().unwrap();
            assert!(!hops.is_empty());
            assert!(hops.iter().any(|hop| hop["alive_voice_count"] == 1));
            let mut total_allocations = 0;
            for (idx, hop) in hops.iter().enumerate() {
                assert_eq!(hop["frame_idx"], idx as u64);
                assert!(hop["alive_voice_count"].as_u64().unwrap() <= 1);
                let expected_time = idx as f64 * 512.0 / 48_000.0;
                assert!((hop["time_sec"].as_f64().unwrap() - expected_time).abs() < 1e-6);
                let elapsed = hop["elapsed_us"].as_f64().unwrap();
                let phases: f64 = [
                    "analysis_wait_us",
                    "listener_wait_us",
                    "landscape_update_us",
                    "population_us",
                    "reports_us",
                    "render_route_us",
                    "post_render_us",
                ]
                .iter()
                .map(|key| {
                    let value = hop[key].as_f64().unwrap();
                    assert!(value.is_finite() && value >= 0.0);
                    value
                })
                .sum();
                assert!(elapsed.is_finite() && elapsed >= phases);
                assert!(
                    hop["synthesis_us"].as_f64().unwrap()
                        <= hop["render_route_us"].as_f64().unwrap()
                );
                assert!(hop["rendered_tone_count"].as_u64().is_some());
                assert!(hop["underrun_frames_total"].is_null());
                if cfg!(feature = "profile-alloc") {
                    total_allocations += hop["worker_allocations"]["count"].as_u64().unwrap();
                    assert!(hop["worker_allocations"]["bytes"].as_u64().is_some());
                } else {
                    assert!(hop["worker_allocations"].is_null());
                }
            }
            if cfg!(feature = "profile-alloc") {
                assert!(total_allocations > 0);
            }
            assert_eq!(profile["summary"]["hop_count"], hops.len());
            let mut elapsed: Vec<f64> = hops
                .iter()
                .map(|row| row["elapsed_us"].as_f64().unwrap())
                .collect();
            elapsed.sort_unstable_by(f64::total_cmp);
            assert_eq!(
                profile["summary"]["elapsed_max_us"].as_f64(),
                elapsed.last().copied()
            );
            assert!(
                profile["summary"]["elapsed_p99_us"].as_f64().unwrap() <= *elapsed.last().unwrap()
            );
            assert_eq!(dir.join("report.jsonl").exists(), report);
            assert!(fs::read_dir(&dir).unwrap().all(|entry| {
                entry
                    .unwrap()
                    .path()
                    .extension()
                    .is_none_or(|ext| ext != "wav")
            }));
            fs::remove_dir_all(dir).unwrap();
        }
    }
}

#[test]
fn profile_creation_failure_exits_nonzero() {
    let dir = fixture(false);
    let output = profile_command(&dir, &dir).output().unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("create profile"));
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn profile_and_report_must_not_share_a_file() {
    let dir = fixture(false);
    let path = dir.join("profile.json");
    let output = profile_command(&dir, &path)
        .arg("--report")
        .arg(&path)
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("profile and report paths must differ")
    );
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn profile_write_failure_exits_nonzero() {
    let dir = fixture(false);
    let output = profile_command(&dir, Path::new("/dev/full"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("profile:"));
    fs::remove_dir_all(dir).unwrap();
}

#[cfg(target_os = "linux")]
#[test]
fn requested_audio_cannot_silently_fall_back_to_device_free_execution() {
    let dir = fixture(false);
    let alsa = dir.join("invalid-alsa.conf");
    fs::write(
        &alsa,
        "pcm.!default { type nonexistent_conchordal_test_plugin }\n",
    )
    .unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_conchordal"))
        .env("ALSA_CONFIG_PATH", alsa)
        .arg(dir.join("scenario.rhai"))
        .arg("--config")
        .arg(dir.join("config.toml"))
        .args(["--nogui", "--play=true"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(String::from_utf8_lossy(&output.stderr).contains("Audio init failed:"));
    fs::remove_dir_all(dir).unwrap();
}
