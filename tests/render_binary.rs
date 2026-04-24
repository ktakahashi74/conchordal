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
let silent = create(sine.pitch_mode("lock"), 1).freq(220.0).amp(0.0);
let audible = create(sine.pitch_mode("lock"), 1).freq(330.0).amp(0.25);
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

    let _ = fs::remove_file(&scenario);
    let _ = fs::remove_file(&wav_path);
}

#[test]
fn conchordal_render_fails_when_wav_thread_panics() {
    let scenario = write_inline_scenario(
        r#"
let tone = create(sine.pitch_mode("lock"), 1).freq(220.0).amp(0.2);
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
