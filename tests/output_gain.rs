use std::fs;
use std::process::Command;

#[test]
fn output_gain_preserves_ecology_and_listener_reports() {
    let directory = std::env::temp_dir().join(format!(
        "conchordal-output-gain-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir(&directory).unwrap();
    let script = directory.join("gain.rhai");
    fs::write(
        &script,
        r#"
seed(21);
let habitat = place(sine().amp(0.02).sustain().anchor().send(habitat_bus), at(330.0));
let presentation = place(sine().amp(0.04).sustain().anchor().send(presentation_bus), at(440.0));
wait(0.5);
release(habitat);
release(presentation);
wait(0.5);
"#,
    )
    .unwrap();
    let mut audio = Vec::new();
    let mut reports = Vec::new();
    for gain_db in [0.0, 6.0] {
        let config = directory.join(format!("gain-{gain_db}.toml"));
        let wav = directory.join(format!("gain-{gain_db}.wav"));
        let report = directory.join(format!("gain-{gain_db}.jsonl"));
        fs::write(
            &config,
            format!("[audio]\noutput_gain_db = {gain_db}\n[analysis]\nnfft = 2048\n"),
        )
        .unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
            .arg(&script)
            .args(["--config", config.to_str().unwrap()])
            .args(["-o", wav.to_str().unwrap()])
            .args(["--report", report.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        audio.push(
            hound::WavReader::open(wav)
                .unwrap()
                .into_samples::<i16>()
                .collect::<Result<Vec<_>, _>>()
                .unwrap(),
        );
        let records: Vec<serde_json::Value> = fs::read_to_string(report)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|row: &serde_json::Value| row["type"] != "hop_timing")
            .collect();
        for kind in ["habituation_scan", "rhythm_observation", "listener_state"] {
            assert!(
                records.iter().any(|row| row["type"] == kind),
                "missing {kind}"
            );
        }
        reports.push(records);
    }
    assert_eq!(
        reports[0], reports[1],
        "ecology and listener must hear unchanged buses"
    );
    assert_eq!(audio[0].len(), audio[1].len());
    assert!(audio[0].iter().any(|&sample| sample != 0));
    let gain = 10.0f64.powf(6.0 / 20.0);
    for (&before, &after) in audio[0].iter().zip(&audio[1]) {
        // Each WAV truncates once to i16, bounding the scaled difference by gain + 1.
        assert!((f64::from(after) - f64::from(before) * gain).abs() <= gain + 1.0);
    }
    fs::remove_dir_all(directory).unwrap();
}
