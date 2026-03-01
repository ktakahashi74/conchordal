use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_wav_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system time before epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "conchordal-render-it-{}-{nanos}.wav",
        std::process::id()
    ))
}

#[test]
fn conchordal_render_generates_valid_wav() {
    let exe = env!("CARGO_BIN_EXE_conchordal-render");
    let scenario = "tests/scripts/minimal_spawn_run.rhai";
    let wav_path = unique_wav_path();
    let wav_path_str = wav_path.to_string_lossy().to_string();

    let output = Command::new(exe)
        .arg(scenario)
        .args(["-o", &wav_path_str])
        .output()
        .unwrap_or_else(|e| panic!("failed to run conchordal-render: {e}"));

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

    let _ = fs::remove_file(&wav_path);
}
