use std::{fs, process::Command};

#[test]
fn zero_support_preferences_preserve_audio_choices_and_events() {
    let directory =
        std::env::temp_dir().join(format!("conchordal-hv3-zero-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let script = directory.join("scene.rhai");
    let config = directory.join("config.toml");
    fs::write(&config, "[analysis]\nnfft = 2048\n").unwrap();
    let source = r#"
seed(735);
place(sine().amp(0.02).sustain().leave_self_out(true), consonance(220, 880).count(2));
wait(0.5);
place(sine().amp(0.02), consonance(220, 880).count(2));
wait(0.5);
"#;
    let mut baseline = None;
    for (case, prefix) in [
        ("omitted", ""),
        (
            "zero",
            "set_support_focus_preference(0); set_support_conflict_preference(0.0); set_related_novelty_preference(0);",
        ),
    ] {
        fs::write(&script, format!("{prefix}\n{source}")).unwrap();
        let wav = directory.join(format!("{case}.wav"));
        let report = directory.join(format!("{case}.jsonl"));
        let result = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
            .arg(&script)
            .arg("--config")
            .arg(&config)
            .arg("-o")
            .arg(&wav)
            .arg("--report")
            .arg(&report)
            .env("RUST_LOG", "warn")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}; evidence {}",
            String::from_utf8_lossy(&result.stderr),
            directory.display()
        );
        let audio = fs::read(&wav).unwrap();
        let records: Vec<serde_json::Value> = fs::read_to_string(&report)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .filter(|row: &serde_json::Value| row["type"] != "hop_timing")
            .collect();
        if let Some((before_audio, before_records)) = &baseline {
            assert_eq!(
                *before_audio, audio,
                "zero support weights must preserve PCM bits"
            );
            assert_eq!(
                *before_records, records,
                "zero support weights must preserve choices and emitted events"
            );
        } else {
            baseline = Some((audio, records));
        }
    }
    fs::remove_dir_all(directory).unwrap();
}
