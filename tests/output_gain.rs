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
    let source = r#"
seed(21);
let habitat = place(sine().amp(0.02).sustain().anchor().send(habitat_bus), at(330.0));
let presentation = place(sine().amp(0.04).sustain().anchor().send(presentation_bus), at(440.0));
wait(0.5);
release(habitat);
release(presentation);
wait(0.5);
"#;
    let mut audio = Vec::new();
    let mut reports = Vec::new();
    let cases = [
        ("config-zero", 0.0, None),
        ("config-six", 6.0, None),
        ("default", 4.5, None),
        ("scene-zero", 4.5, Some(0.0)),
        ("scene-six", 4.5, Some(6.0)),
        ("scene-minus-six", 4.5, Some(-6.0)),
        ("scene-mixed", 2.0, Some(2.5)),
    ];
    for (name, gain_db, scene_db) in cases {
        let declaration = scene_db
            .map(|db| format!("output_gain({db});\n"))
            .unwrap_or_default();
        fs::write(&script, format!("{declaration}{source}")).unwrap();
        let config = directory.join(format!("{name}.toml"));
        let wav = directory.join(format!("{name}.wav"));
        let report = directory.join(format!("{name}.jsonl"));
        let audio_config = if gain_db == 4.5 {
            String::new()
        } else {
            format!("[audio]\noutput_gain_db = {gain_db}\n")
        };
        fs::write(&config, format!("{audio_config}[analysis]\nnfft = 2048\n")).unwrap();
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
    for report in &reports[1..] {
        assert_eq!(
            reports[0], *report,
            "ecology and listener must hear unchanged buses"
        );
    }
    assert_eq!(
        audio[2], audio[3],
        "explicit scene 0 dB preserves output bits"
    );
    assert_eq!(audio[2], audio[6], "config and scene gains add in dB");
    assert!(audio[0].iter().any(|&sample| sample != 0));
    for (index, (_, config_db, scene_db)) in cases.iter().enumerate().skip(1) {
        let gain = 10.0f64.powf((config_db + scene_db.unwrap_or(0.0)) / 20.0);
        assert_eq!(audio[0].len(), audio[index].len());
        for (&before, &after) in audio[0].iter().zip(&audio[index]) {
            // Each WAV truncates once to i16, bounding the scaled difference by gain + 1.
            assert!((f64::from(after) - f64::from(before) * gain).abs() <= gain + 1.0);
        }
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn combined_output_gain_rejects_unrepresentable_amplitudes_at_the_declaration() {
    let directory =
        std::env::temp_dir().join(format!("conchordal-combined-gain-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let script = directory.join("gain.rhai");
    let config = directory.join("gain.toml");
    let wav = directory.join("gain.wav");
    for (config_db, scene_db) in [(760.0, 100.0), (-880.0, -100.0)] {
        fs::write(&config, format!("[audio]\noutput_gain_db = {config_db}\n")).unwrap();
        fs::write(&script, format!("\noutput_gain({scene_db});\nwait(0.1);")).unwrap();
        let result = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
            .arg(&script)
            .arg("--config")
            .arg(&config)
            .arg("-o")
            .arg(&wav)
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(!result.status.success(), "{error}");
        assert!(error.contains("gain.rhai:2:1:"), "{error}");
        assert!(
            error.contains("output_gain + audio.output_gain_db"),
            "{error}"
        );
        assert!(
            !wav.exists(),
            "invalid combined gain must fail before opening output"
        );
    }
    fs::remove_dir_all(directory).unwrap();
}
