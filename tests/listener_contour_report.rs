use std::fs;
use std::process::Command;

use serde_json::Value;

#[test]
fn passive_contour_report_uses_presentation_audio_and_preserves_rendered_samples() {
    let root = std::path::PathBuf::from(format!(
        "target/listener-contour-report-{}",
        std::process::id()
    ));
    fs::create_dir_all(&root).unwrap();
    let config = root.join("config.toml");
    fs::write(&config, "[dcc]\ncoupling_strength = 0.0\n").unwrap();
    let report = root.join("report.jsonl");
    for enabled in [false, true] {
        let mut command = Command::new(env!("CARGO_BIN_EXE_conchordal-render"));
        command
            .args(["tests/scripts/listener_contour.rhai", "--config"])
            .arg(&config)
            .arg("--output")
            .arg(root.join(format!("report-{enabled}.wav")));
        if enabled {
            command.arg("--report").arg(&report);
        }
        let output = command.output().unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    assert_eq!(
        fs::read(root.join("report-false.wav")).unwrap(),
        fs::read(root.join("report-true.wav")).unwrap()
    );
    let records: Vec<Value> = fs::read_to_string(&report)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let episodes: Vec<_> = records
        .iter()
        .filter(|r| r["type"] == "listener_contour" && r["event"] == "periodic_episode")
        .collect();
    assert_eq!(episodes.len(), 56);
    let pitches = [220.0, 252.7, 311.1, 274.6, 359.8, 335.9, 239.1, 294.3];
    for (index, row) in episodes.iter().enumerate() {
        let pitch = row["periodic_frequency_hz"].as_f64().unwrap();
        assert!(
            (1200.0 * (pitch / pitches[index % 8]).log2()).abs() < 3.0,
            "{row}"
        );
        let onset = row["onset_sec"].as_f64().unwrap();
        let available = row["time_sec"].as_f64().unwrap();
        assert!(onset >= 0.6 && available > onset && available - onset < 0.15);
        assert!(row.get("voice_id").is_none());
        assert!(row.get("analysis_frame_id").is_none());
        if index > 32 {
            assert!(!row["error_candidate"].as_bool().unwrap());
            assert!(row["gain_bits"].as_f64().unwrap() > 0.0);
        }
    }
    assert!(episodes[0]["delta_log2"].is_null());
    assert!(
        records
            .iter()
            .any(|r| r["type"] == "listener_contour" && r["event"] == "silence_gap")
    );
    assert!(
        !records
            .iter()
            .any(|r| r["type"] == "listener_contour" && r["event"] == "input_gap")
    );
    fs::remove_dir_all(root).unwrap();
}
