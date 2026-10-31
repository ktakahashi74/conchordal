use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicU64, Ordering};

#[path = "support/test_output_directory.rs"]
mod test_output_directory;
use test_output_directory::TestOutputDirectory;

fn fixture() -> (PathBuf, TestOutputDirectory) {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "conchordal-open-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let cleanup = TestOutputDirectory::new(dir.clone());
    fs::write(dir.join("config.toml"), "[analysis]\nnfft = 2048\n").unwrap();
    (dir, cleanup)
}

fn instrument(dir: &Path, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_conchordal"))
        .env("RUST_LOG", "warn")
        .arg(dir.join("input.rhai"))
        .args(["--nogui", "--play=false", "--config"])
        .arg(dir.join("config.toml"))
        .args(arguments)
        .output()
        .unwrap()
}

fn render(dir: &Path, name: &str, arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .env("RUST_LOG", "warn")
        .arg(dir.join("input.rhai"))
        .arg("--config")
        .arg(dir.join("config.toml"))
        .arg("--output")
        .arg(dir.join(name))
        .args(arguments)
        .output()
        .unwrap()
}

#[test]
fn open_compile_only_and_cli_bounds_are_explicit() {
    let (dir, _cleanup) = fixture();
    fs::write(
        dir.join("input.rhai"),
        "seed(21); endless(); place(sine().sustain(), at(220));",
    )
    .unwrap();
    let unbounded = instrument(&dir, &["--compile-only"]);
    assert!(!unbounded.status.success());
    assert!(String::from_utf8_lossy(&unbounded.stderr).contains("requires --duration"));
    let compiled = instrument(&dir, &["--compile-only", "--duration", "1"]);
    assert!(compiled.status.success(), "{:?}", compiled);
    assert!(String::from_utf8_lossy(&compiled.stderr).contains("end: open"));
    for (arguments, message) in [
        (
            vec!["--report", "unused.jsonl"],
            "--report with an open scenario requires --duration",
        ),
        (
            vec!["--profile", "unused.json"],
            "--profile with an open scenario requires --duration",
        ),
        (
            vec!["--duration", "2000", "--profile", "unused.json"],
            "PROFILE_HOP_CAPACITY",
        ),
    ] {
        let output = instrument(&dir, &arguments);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(message),
            "{:?}",
            output
        );
    }
    for arguments in [vec![], vec!["--duration", "50000"]] {
        let output = render(&dir, "rejected.wav", &arguments);
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(if arguments.is_empty() {
                "require --duration"
            } else {
                "WAV size limit"
            }),
            "{:?}",
            output
        );
        assert!(!dir.join("rejected.wav").exists());
    }
    for value in ["--duration=-1", "--duration=NaN", "--duration=inf"] {
        for output in [
            instrument(&dir, &[value]),
            render(&dir, "invalid.wav", &[value]),
        ] {
            assert!(!output.status.success());
            assert!(
                String::from_utf8_lossy(&output.stderr)
                    .contains("duration must be finite and non-negative")
            );
        }
    }
    let report = dir.join("bounded.jsonl");
    let profile = dir.join("bounded.json");
    let output = instrument(
        &dir,
        &[
            "--duration",
            "0.04",
            "--report",
            report.to_str().unwrap(),
            "--profile",
            profile.to_str().unwrap(),
        ],
    );
    assert!(output.status.success(), "{:?}", output);
    assert!(fs::metadata(report).unwrap().len() > 0);
    let profile: serde_json::Value = serde_json::from_slice(&fs::read(profile).unwrap()).unwrap();
    assert!(profile.is_object());
    assert!(!Path::new("unused.jsonl").exists());
    assert!(!Path::new("unused.json").exists());
}

#[test]
fn finite_script_return_type_check_is_preserved() {
    let (dir, _cleanup) = fixture();
    fs::write(dir.join("input.rhai"), "seed(21); place(sine(), at(220));").unwrap();
    let output = instrument(&dir, &["--compile-only"]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("Output type incorrect: Population (expecting ())")
    );
}

#[test]
fn duration_keeps_the_release_tail_and_is_deterministic() {
    let (dir, _cleanup) = fixture();
    let body = "place(sine().anchor().sustain().amp(0.1).adsr(0,0,1,0.1), at(220));";
    fs::write(
        dir.join("input.rhai"),
        format!("seed(21); endless(); {body}"),
    )
    .unwrap();
    for name in ["open-a.wav", "open-b.wav"] {
        let output = render(&dir, name, &["--duration", "0.08"]);
        assert!(output.status.success(), "{:?}", output);
    }
    let first = fs::read(dir.join("open-a.wav")).unwrap();
    assert_eq!(first, fs::read(dir.join("open-b.wav")).unwrap());
    let mut reader = hound::WavReader::open(dir.join("open-a.wav")).unwrap();
    let rate = reader.spec().sample_rate;
    assert!(reader.duration() >= (rate as f64 * 0.18) as u32);
    assert!(
        reader.duration() < rate,
        "a pending Finish or missing end must not extend the run"
    );
    let samples: Vec<_> = reader.samples::<i16>().map(Result::unwrap).collect();
    let fade_start = (rate as f64 * 0.10) as usize;
    let fade_end = (rate as f64 * 0.14) as usize;
    assert!(
        samples[fade_start..fade_end]
            .iter()
            .any(|sample| *sample != 0),
        "the Voice must sound during its fade"
    );
    fs::write(
        dir.join("input.rhai"),
        format!("seed(21); {body} wait(100);"),
    )
    .unwrap();
    let output = render(&dir, "finite-cut.wav", &["--duration", "0.08"]);
    assert!(output.status.success(), "{:?}", output);
    assert_eq!(first, fs::read(dir.join("finite-cut.wav")).unwrap());
}

#[test]
fn finite_render_without_duration_keeps_its_normal_finish() {
    let (dir, _cleanup) = fixture();
    fs::write(dir.join("input.rhai"), "seed(21); play(|| { place(sine().anchor().sustain().adsr(0,0,1,0.03), at(220)); wait(0.04); });").unwrap();
    for (name, args) in [
        ("normal.wav", vec![]),
        ("later.wav", vec!["--duration", "10"]),
    ] {
        let output = render(&dir, name, &args);
        assert!(output.status.success(), "{:?}", output);
    }
    assert_eq!(
        fs::read(dir.join("normal.wav")).unwrap(),
        fs::read(dir.join("later.wav")).unwrap()
    );
}

#[cfg(unix)]
#[test]
fn first_ctrl_c_fades_and_second_ctrl_c_exits_immediately() {
    use std::process::Stdio;
    use std::time::{Duration, Instant};

    for second in [false, true] {
        let (dir, _cleanup) = fixture();
        fs::write(
            dir.join("input.rhai"),
            format!(
                "seed(21); endless(); place(sine().anchor().sustain().adsr(0,0,1,{}), at(220));",
                if second { 3600.0 } else { 0.1 }
            ),
        )
        .unwrap();
        let stdout = dir.join("stdout.log");
        let stderr = dir.join("stderr.log");
        let mut child = Command::new(env!("CARGO_BIN_EXE_conchordal"))
            .env("RUST_LOG", "info")
            .arg(dir.join("input.rhai"))
            .args(["--nogui", "--play=false", "--config"])
            .arg(dir.join("config.toml"))
            .stdout(fs::File::create(&stdout).unwrap())
            .stderr(Stdio::from(fs::File::create(&stderr).unwrap()))
            .spawn()
            .unwrap();
        let deadline = Instant::now() + Duration::from_secs(10);
        while !fs::read_to_string(&stdout)
            .unwrap()
            .contains("Event: Spawn")
            && child.try_wait().unwrap().is_none()
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(2));
        }
        if child.try_wait().unwrap().is_some() || Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            panic!(
                "instrument did not start: {}",
                fs::read_to_string(stderr).unwrap()
            );
        }
        assert!(
            Command::new("kill")
                .args(["-INT", &child.id().to_string()])
                .status()
                .unwrap()
                .success()
        );
        while !fs::read_to_string(&stderr)
            .unwrap()
            .contains("ending (fade)…")
            && Instant::now() < deadline
        {
            std::thread::sleep(Duration::from_millis(2));
        }
        if second {
            std::thread::sleep(Duration::from_millis(20));
            assert!(
                child.try_wait().unwrap().is_none(),
                "first Ctrl-C must preserve the long fade"
            );
            assert!(
                Command::new("kill")
                    .args(["-INT", &child.id().to_string()])
                    .status()
                    .unwrap()
                    .success()
            );
        }
        while child.try_wait().unwrap().is_none() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(2));
        }
        let result = child.try_wait().unwrap();
        if result.is_none() {
            let _ = child.kill();
        }
        let status = child.wait().unwrap();
        assert!(
            result.is_some() && status.success(),
            "Ctrl-C must end the instrument: {}",
            fs::read_to_string(&stderr).unwrap()
        );
        assert!(
            fs::read_to_string(stderr)
                .unwrap()
                .contains("ending (fade)…")
        );
    }
}

#[test]
fn incremental_render_and_compile_overflow_at_the_same_score_hop() {
    for source in [
        "endless(); wait(0.02); loop { set_global_coupling(0.25); }",
        "endless(); wait(0.02); let name = \"x\"; loop { section(name, || {}); name += name; }",
    ] {
        let (dir, _cleanup) = fixture();
        fs::write(dir.join("input.rhai"), source).unwrap();
        for output in [
            instrument(&dir, &["--compile-only", "--duration", "0.04"]),
            render(&dir, "overflow.wav", &["--duration", "0.04"]),
        ] {
            assert!(!output.status.success());
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                stderr.contains(
                    "score items within one hop exceed the ring capacity at t=0.021333333"
                ),
                "{stderr}"
            );
        }
    }
}

#[test]
fn looping_endless_render_reports_repeat_at_duration_bound() {
    let (dir, _cleanup) = fixture();
    fs::write(
        dir.join("input.rhai"),
        r#"
        seed(21); endless();
        section("loop", || {
            let p = place(sine().anchor().sustain().adsr(0,0,1,0.02), at(220));
            loop { p.amp(0.1); wait(0.02); }
        });
    "#,
    )
    .unwrap();
    let mut reports = Vec::new();
    for name in ["a", "b"] {
        let report = dir.join(format!("{name}.jsonl"));
        let output = render(
            &dir,
            &format!("{name}.wav"),
            &["--duration", "0.08", "--report", report.to_str().unwrap()],
        );
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let rows: Vec<serde_json::Value> = fs::read_to_string(report)
            .unwrap()
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
        reports.push(rows);
    }
    assert_eq!(reports[0].len(), reports[1].len());
    assert!(reports[0].iter().any(|r| r["type"] == "scene_marker"));
    for (a, b) in reports[0].iter().zip(&reports[1]) {
        assert_eq!(a, b);
    }
}

#[test]
fn oversized_item_after_header_is_reported_and_ecology_continues() {
    let (dir, _cleanup) = fixture();
    // 128 MiB exceeds the calibrated 77,594,624-byte payload capacity.
    fs::write(
        dir.join("input.rhai"),
        r#"
        endless(); place(sine().sustain(), at(220)); wait(0.02);
        let name = "x"; for i in 0..27 { name += name; }
        section(name, || {});
    "#,
    )
    .unwrap();
    let compiled = instrument(&dir, &["--compile-only", "--duration", "0.04"]);
    assert!(!compiled.status.success());
    assert!(
        String::from_utf8_lossy(&compiled.stderr).contains("score item exceeds the ring capacity")
    );
    let rendered = render(&dir, "failed-script.wav", &["--duration", "0.04"]);
    assert!(
        rendered.status.success(),
        "{}",
        String::from_utf8_lossy(&rendered.stderr)
    );
    assert!(
        String::from_utf8_lossy(&rendered.stderr).contains("score item exceeds the ring capacity")
    );
    assert!(fs::metadata(dir.join("failed-script.wav")).unwrap().len() > 44);
}
