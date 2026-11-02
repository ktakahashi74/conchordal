use std::{fs, process::Command};

#[test]
fn zero_harmonic_relations_preserves_audio_choices_and_events_with_existing_axes() {
    let directory =
        std::env::temp_dir().join(format!("conchordal-hv7-zero-{}", std::process::id()));
    fs::create_dir(&directory).unwrap();
    let config = directory.join("config.toml");
    let script = directory.join("scene.rhai");
    fs::write(&config, "[analysis]\nnfft = 2048\n").unwrap();
    // Frequencies are selected by the ecology; the worker supplies ranges and bodies.
    let source = r#"
seed(735);
CONTROL
AXES
place(sine().amp(0.02).sustain().leave_self_out(true).temperature(0.3), consonance(220, 880).count(2));
wait(0.25);
LIVE_CONTROL
place(harmonic().amp(0.02).sustain().leave_self_out(true).pitch_core("peak_sampler"), consonance(220, 880).peak());
place(flora().amp(0.02).fade(0), consonance(220, 880));
wait(0.25);
place(sine().amp(0.02), consonance(220, 880).tension(0.4));
wait(0.25);
"#;
    for (name, axes) in [
        ("base", ""),
        (
            "vertical_and_existing",
            "set_support_focus_preference(1); set_support_conflict_preference(0.5); set_related_novelty_preference(1); set_roughness_aversion(0.8); set_harmonicity_preference(1.2); harmonic_rhythm(2);",
        ),
    ] {
        let mut baseline = None;
        for (case, initial, live) in [
            ("omitted", "", ""),
            (
                "zero",
                "follow_harmonic_relations(0); follow_harmonic_continuation(0);",
                "follow_harmonic_relations(0.0); follow_harmonic_continuation(0.0);",
            ),
        ] {
            fs::write(
                &script,
                source
                    .replace("LIVE_CONTROL", live)
                    .replace("CONTROL", initial)
                    .replace("AXES", axes),
            )
            .unwrap();
            let wav = directory.join(format!("{name}-{case}.wav"));
            let report = directory.join(format!("{name}-{case}.jsonl"));
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
                "{name}/{case}: {}; evidence {}",
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
                    "{name}: zero must preserve all WAV bytes"
                );
                assert_eq!(
                    *before_records, records,
                    "{name}: zero must preserve all non-timing event records"
                );
            } else {
                baseline = Some((audio, records));
            }
        }
    }
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn positive_harmonic_relations_sparse_opportunity_compatibility_smoke() {
    // The worker authors fixed 440/220 Hz carriers, not ecology-generated harmony.
    // Shared harmonic rhythm deliberately makes legal opportunities sparse here.
    // This is bounded runtime compatibility, not declared-load or causal acceptance;
    // tests/research/harmonic_relations_load.rhai retains the full declared load.
    let directory = std::env::temp_dir().join(format!(
        "conchordal-hv7-sparse-positive-{}",
        std::process::id()
    ));
    fs::create_dir(&directory).unwrap();
    let config = directory.join("config.toml");
    let script = directory.join("scene.rhai");
    let wav = directory.join("audio.wav");
    let report = directory.join("report.jsonl");
    fs::write(&config, "[analysis]\nnfft = 2048\n").unwrap();
    fs::write(
        &script,
        r#"
// Worker-authored fixed carriers; the ecology did not generate their relationship.
// Sparse legal opportunities for compatibility only, not the declared-load assay.
seed(735);
follow_harmonic_relations(0.8);
harmonic_rhythm(2);
set_support_focus_preference(1.0);
set_support_conflict_preference(0.5);
set_related_novelty_preference(1.0);
place(sine().amp(0.02).sustain().anchor(), at(440.0));
place(harmonic().amp(0.02).sustain().anchor(), at(220.0));
wait(0.5);
place(modal().amp(0.02).sustain().leave_self_out(true), consonance(220.0, 880.0).peak().spacing(0));
place(harmonic().amp(0.02).sustain().leave_self_out(true), consonance(220.0, 880.0));
place(flora().amp(0.02).fade(0).tone("sine"), consonance(220.0, 880.0));
place(flora().amp(0.02).fade(0).tone("resonant"), consonance(220.0, 880.0));
wait(0.5);
"#,
    )
    .unwrap();
    let result = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
        .arg(&script)
        .arg("--config")
        .arg(&config)
        .arg("-o")
        .arg(&wav)
        .arg("--report")
        .arg(&report)
        .env("RUST_LOG", "warn,conchordal::runtime::worker=info")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&result.stderr);
    assert!(
        result.status.success(),
        "{stderr}; evidence {}",
        directory.display()
    );
    let stdout = String::from_utf8_lossy(&result.stdout);
    assert!(
        stdout.contains("harmonic relations runtime")
            || stderr.contains("harmonic relations runtime"),
        "temporal worker must run: {stdout} {stderr}"
    );
    assert!(fs::metadata(wav).unwrap().len() > 44);
    let records: Vec<serde_json::Value> = fs::read_to_string(report)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(!records.is_empty());
    fs::remove_dir_all(directory).unwrap();
}
