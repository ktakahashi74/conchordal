use std::fs;
use std::process::Command;

#[test]
fn phase3_finish_waits_for_modal_tail_but_not_oscillator_tails() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "conchordal-phase3-finish-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&dir).unwrap();
    let config = dir.join("config.toml");
    fs::write(&config, "render_prototype = true\n[analysis]\nnfft = 2048\nhop_size = 512\n[psychoacoustics.habituation]\nenabled = false\n[dcc]\ncoupling_strength = 0.0\n").unwrap();
    let mut lengths = Vec::new();
    for body in ["sine", "harmonic", "modal"] {
        let script = dir.join(format!("{body}.rhai"));
        fs::write(
            &script,
            format!(
                r#"
let a = place({body}().amp(0.05).sustain().anchor().adsr(0.01, 0.0, 1.0, 0.01), at(220.0));
wait(0.1);
release(a);
wait(0.03);
"#
            ),
        )
        .unwrap();
        let wav = dir.join(format!("{body}.wav"));
        let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
            .arg(&script)
            .arg("--config")
            .arg(&config)
            .arg("--seed")
            .arg("73")
            .arg("-o")
            .arg(&wav)
            .env("RUST_LOG", "warn")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let pcm: Vec<i16> = hound::WavReader::open(&wav)
            .unwrap()
            .samples::<i16>()
            .map(Result::unwrap)
            .collect();
        assert!(pcm.iter().any(|v| *v != 0));
        if body != "modal" {
            assert_eq!(pcm.last(), Some(&0));
        }
        let profile = dir.join(format!("{body}-instrument.json"));
        let instrument = Command::new(env!("CARGO_BIN_EXE_conchordal"))
            .arg(&script)
            .arg("--config")
            .arg(&config)
            .args(["--nogui", "--play=false", "--seed", "73", "--profile"])
            .arg(&profile)
            .env("RUST_LOG", "warn")
            .output()
            .unwrap();
        assert!(
            instrument.status.success(),
            "{}",
            String::from_utf8_lossy(&instrument.stderr)
        );
        let instrument: serde_json::Value =
            serde_json::from_slice(&fs::read(profile).unwrap()).unwrap();
        assert_eq!(instrument["audio_output"], "no_device");
        assert_eq!(
            instrument["summary"]["hop_count"].as_u64().unwrap() as usize * 512,
            pcm.len(),
            "instrument Finish differs from offline disposal for {body}"
        );
        lengths.push(pcm.len());
    }
    assert!(
        lengths[2] > lengths[0] && lengths[2] > lengths[1],
        "Finish must retain Modal radiation and retire zero-residual oscillators: {lengths:?}"
    );
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn phase3_binary_is_seeded_and_reports_body_capabilities_with_real_audio() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir =
        std::env::temp_dir().join(format!("conchordal-phase3-{}-{stamp}", std::process::id()));
    fs::create_dir(&dir).unwrap();
    let config = dir.join("config.toml");
    fs::write(
        &config,
        r#"
render_prototype = true
[analysis]
nfft = 2048
hop_size = 512
[psychoacoustics.habituation]
enabled = false
[dcc]
coupling_strength = 0.0
[temporal_body]
means = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
deviations = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
accent_means = [0.0, 0.0]
accent_deviations = [1.0, 1.0]
[temporal_onset_comparison]
footprint = "body"
"#,
    )
    .unwrap();
    let script = dir.join("bounded.rhai");
    fs::write(
        &script,
        r#"
temporal_mode("observe");
let a = place(sine().amp(0.05).sustain().anchor().adsr(0.01, 0.0, 1.0, 0.01), at(220.0));
let b = place(harmonic().amp(0.03).sustain().anchor().adsr(0.01, 0.0, 1.0, 0.01), at(330.0));
let c = place(modal().amp(0.03).sustain().anchor().adsr(0.01, 0.0, 1.0, 0.01), at(440.0));
let d = place(sine().amp(0.02).flow().cycles(3).anchor().adsr(0.01, 0.0, 1.0, 0.01), at(230.0));
let e = place(harmonic().amp(0.02).flow().cycles(3).anchor().adsr(0.01, 0.0, 1.0, 0.01), at(340.0));
let f = place(modal().amp(0.02).flow().cycles(3).anchor().adsr(0.01, 0.0, 1.0, 0.01), at(450.0));
wait(0.3);
release(a); release(b); release(c);
release(d); release(e); release(f);
wait(0.1);
"#,
    )
    .unwrap();
    let mut baseline = None;
    let mut baseline_schedule = None;
    for run in 0..2 {
        let wav = dir.join(format!("run-{run}.wav"));
        let report = dir.join(format!("run-{run}.jsonl"));
        let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
            .arg(&script)
            .arg("--config")
            .arg(&config)
            .arg("--seed")
            .arg("73")
            .arg("-o")
            .arg(&wav)
            .arg("--report")
            .arg(&report)
            .env("RUST_LOG", "warn")
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let pcm: Vec<i16> = hound::WavReader::open(&wav)
            .unwrap()
            .samples::<i16>()
            .map(Result::unwrap)
            .collect();
        assert!(
            pcm.len() > 48_000 / 2,
            "Finish cut off the retained free tails"
        );
        assert!(pcm.iter().any(|v| *v != 0));
        assert!(pcm.iter().all(|v| v.unsigned_abs() < i16::MAX as u16));
        let records: Vec<serde_json::Value> = fs::read_to_string(&report)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let schedule: Vec<_> = records
            .iter()
            .filter(|row| {
                matches!(
                    row["type"].as_str(),
                    Some("spawn" | "respawn" | "death" | "onset" | "scene_marker")
                )
            })
            .cloned()
            .collect();
        assert!(!schedule.is_empty());
        let footprints: Vec<_> = records
            .iter()
            .filter(|row| row["type"] == "body_footprint")
            .collect();
        assert!(!footprints.is_empty());
        for row in &footprints {
            let source = row["identity"]["source_id"].as_u64().unwrap();
            if matches!(source, 3 | 6) {
                assert_eq!(row["state"]["unsupported"], "renderer-phase3");
                assert_eq!(row["d_samples"], 0);
            } else {
                assert!(matches!(source, 1 | 2 | 4 | 5));
                assert_eq!(row["state"], "body");
                assert!(row["d_samples"].as_u64().unwrap() > 0);
            }
        }
        for source in [4, 5, 6] {
            assert!(
                footprints
                    .iter()
                    .any(|row| row["identity"]["source_id"] == source)
            );
        }
        if let Some(expected) = &baseline {
            assert_eq!(&pcm, expected);
        }
        if let Some(expected) = &baseline_schedule {
            assert_eq!(&schedule, expected);
        }
        baseline = Some(pcm);
        baseline_schedule = Some(schedule);
    }
    fs::remove_dir_all(dir).unwrap();
}

#[test]
fn phase3_modal_field_birth_keeps_terrain_placement() {
    let stamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "conchordal-phase3-birth-{}-{stamp}",
        std::process::id()
    ));
    fs::create_dir(&dir).unwrap();
    let script = dir.join("field.rhai");
    fs::write(&script, r#"
let terrain = place(sine().amp(0.2).sustain().anchor(), at(440.0));
wait(1.5);
let c = place(modal().modes(custom_modes([1.0, 2.9])).brightness(1.0).amp(0.01).sustain().anchor(), consonance(180.0, 500.0).peak().count(3));
wait(0.15);
release(c); release(terrain);
wait(0.1);
"#).unwrap();
    let mut legacy_births = Vec::new();
    let mut prototype_control = None;
    for prototype in [false, true] {
        for surrogate in [false, true] {
            let config = dir.join("config.toml");
            fs::write(&config, format!(
                "birth_surrogate = {surrogate}\n{}[analysis]\nnfft = 2048\nhop_size = 512\n[psychoacoustics.habituation]\nenabled = false\n[dcc]\ncoupling_strength = 0.0\n",
                if prototype { "render_prototype = true\n" } else { "" }
            )).unwrap();
            let wav = dir.join("render.wav");
            let report = dir.join("report.jsonl");
            let output = Command::new(env!("CARGO_BIN_EXE_conchordal-render"))
                .arg(&script)
                .arg("--config")
                .arg(&config)
                .arg("--seed")
                .arg("73")
                .arg("-o")
                .arg(&wav)
                .arg("--report")
                .arg(&report)
                .env("RUST_LOG", "warn")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            let births: Vec<serde_json::Value> = fs::read_to_string(&report)
                .unwrap()
                .lines()
                .map(|line| serde_json::from_str::<serde_json::Value>(line).unwrap())
                .filter(|row| row["type"] == "spawn")
                .collect();
            assert_eq!(births.len(), 4);
            if prototype {
                let result = (births, fs::read(&wav).unwrap());
                if let Some(control) = &prototype_control {
                    assert_eq!(
                        &result, control,
                        "native Modal must retain terrain-only placement"
                    );
                } else {
                    prototype_control = Some(result);
                }
            } else {
                legacy_births.push(births);
            }
        }
    }
    assert_ne!(
        legacy_births[0], legacy_births[1],
        "the fixture must exercise supported legacy Field birth"
    );
    fs::remove_dir_all(dir).unwrap();
}
